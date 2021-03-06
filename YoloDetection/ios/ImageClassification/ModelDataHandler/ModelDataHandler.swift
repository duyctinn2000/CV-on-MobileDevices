// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

/// A result from invoking the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let inferences: [Inference]
}

/// An inference from invoking the `Interpreter`.
struct Inference {
  let rect: CGRect
  let confidence: Float
  let className: String
}

struct Heap<Element> {
  var elements : [Element]
  let priorityFunction : (Element, Element) -> Bool

  // TODO: priority queue functions
  // TODO: helper functions
    var isEmpty : Bool {
      return elements.isEmpty
    }

    var count : Int {
      return elements.count
    }
    func peek() -> Element? {
      return elements.first
    }
    func isRoot(_ index: Int) -> Bool {
      return (index == 0)
    }

    func leftChildIndex(of index: Int) -> Int {
      return (2 * index) + 1
    }

    func rightChildIndex(of index: Int) -> Int {
      return (2 * index) + 2
    }

    func parentIndex(of index: Int) -> Int {
      return (index - 1) / 2
    }
    func isHigherPriority(at firstIndex: Int, than secondIndex: Int) -> Bool {
        return priorityFunction(elements[firstIndex], elements[secondIndex])
    }
    func highestPriorityIndex(of parentIndex: Int, and childIndex: Int) -> Int {
      guard childIndex < count && isHigherPriority(at: childIndex, than: parentIndex)
        else { return parentIndex }
      return childIndex
    }
        
    func highestPriorityIndex(for parent: Int) -> Int {
      return highestPriorityIndex(of: highestPriorityIndex(of: parent, and: leftChildIndex(of: parent)), and: rightChildIndex(of: parent))
    }
    mutating func swapElement(at firstIndex: Int, with secondIndex: Int) {
      guard firstIndex != secondIndex
        else { return }
        elements.swapAt(firstIndex, secondIndex)
    }
    mutating func enqueue(_ element: Element) {
      elements.append(element)
      siftUp(elementAtIndex: count - 1)
    }
    mutating func siftUp(elementAtIndex index: Int) {
      let parent = parentIndex(of: index) // 1
      guard !isRoot(index), // 2
        isHigherPriority(at: index, than: parent) // 3
        else { return }
      swapElement(at: index, with: parent) // 4
      siftUp(elementAtIndex: parent) // 5
    }
    mutating func dequeue() -> Element? {
      guard !isEmpty // 1
        else { return nil }
      swapElement(at: 0, with: count - 1) // 2
      let element = elements.removeLast() // 3
      if !isEmpty { // 4
        siftDown(elementAtIndex: 0) // 5
      }
      return element // 6
    }
    mutating func siftDown(elementAtIndex index: Int) {
      let childIndex = highestPriorityIndex(for: index) // 1
      if index == childIndex { // 2
        return
      }
      swapElement(at: index, with: childIndex) // 3
      siftDown(elementAtIndex: childIndex)
    }
}


/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet model.
enum MobileNet {
  static let modelInfo: FileInfo = (name: "yolov5-m", extension: "tflite")
  static let labelsInfo: FileInfo = (name: "labels", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler : NSObject {

  // MARK: - Internal Properties

  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int

  let resultCount = 3
  let threadCountLimit = 10

  // MARK: - Model Parameters
  let threshold: Float = 0.3
  let batchSize = 4
  let inputChannels = 3
  let inputWidth = 320
  let inputHeight = 320
    let mNmsThresh : Float = 0.45
    let output_box = 6300
    
    let numClass: Int = 80

  // MARK: - Private Properties

  /// List of labels from the given labels file.
  private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name
 

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
      var options = Interpreter.Options()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }
      
      super.init()
    // Load the classes listed in the labels file.
    loadLabels(fileInfo: labelsFileInfo)
  }

  // MARK: - Internal Methods

  /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
    
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)


    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize) else {
      return nil
    }

    let interval: TimeInterval
    let outputTensor: Tensor
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        thumbnailPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }

      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      interval = Date().timeIntervalSince(startDate) * 1000

      // Get the output `Tensor` to process the inference results.
      outputTensor = try interpreter.output(at: 0)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    let results: [Float]
    switch outputTensor.dataType {
    case .uInt8:
      guard let quantization = outputTensor.quantizationParameters else {
        print("No results returned because the quantization values for the output tensor are nil.")
        return nil
      }
      let quantizedResults = [UInt8](outputTensor.data)
      results = quantizedResults.map {
        quantization.scale * Float(Int($0) - quantization.zeroPoint)
      }
    case .float32:
      results = [Float32](unsafeData: outputTensor.data) ?? []
    default:
      print("Output tensor data type \(outputTensor.dataType) is unsupported for this example app.")
      return nil
    }
    
    // Process the results.
      var out = [[[Float32]]](repeating: [[Float32]](repeating: [Float32](repeating: -1, count: self.numClass+5), count: output_box), count: 1)
      var i_results = 0
      for i in (0..<self.output_box) {
          
          for j in (0..<self.numClass+5) {
              out[0][i][j] = results[i_results]
              i_results += 1
          }
          
          for j in (0..<4) {
              out[0][i][j] *= Float32(self.inputWidth)
          }
      }
    
      var detections : [Inference] = []
      
      for i in (0..<self.output_box){
          let confidence : Float32 = out[0][i][4];
          var detectedClass = -1;
          var maxClass : Float32 = 0;
          var classes: [Float32] = []

          for  c in (0..<self.numClass) {
              classes.append(out[0][i][5 + c]);
          }

          for c  in (0..<self.numClass) {
              if (classes[c] > maxClass) {
                  detectedClass = c;
                  maxClass = classes[c];
              }
          }

          let confidenceInClass = maxClass * confidence;
          if (confidenceInClass > self.threshold &&  labels[detectedClass]=="dog") {
              let xPos = out[0][i][0];
              let yPos = out[0][i][1];

              let w = out[0][i][2];
              let h = out[0][i][3];
              var rect: CGRect = CGRect.zero
              
              rect.origin.x = CGFloat(max(0,xPos - w / 2))
              rect.origin.y = CGFloat(max(0,yPos - h / 2))
              rect.size.width = CGFloat(min(Float32(self.inputWidth-1),w))
              rect.size.height = CGFloat(min(Float32(self.inputHeight-1),h))

              detections.append(Inference(rect: rect, confidence: confidenceInClass, className: labels[detectedClass]))
          }
      }
      
      print(detections.count)
      let recognitions : [Inference] = self.nms(results: detections)
      print(recognitions)
    
    // Return the inference time and inference results.
    return Result(inferenceTime: interval, inferences: recognitions)
  }

  // MARK: - Private Methods

  /// Returns the top N inference results sorted in descending order.
  private func nms(results: [Inference]) -> [Inference] {
      
      var nmsList : [Inference] = [];
//      var pqList : [Inference] = [];
      var pqList = Heap<Inference>(elements: [], priorityFunction: {
              (a, b) in
          return a.confidence > b.confidence
          })

      for k in (0..<self.numClass) {
          pqList.elements=[]
          //1.find max confidence per class
          
          for i in (0..<results.count) {
              if (results[i].className == labels[k]) {
                  pqList.enqueue(results[i])
              }
              
          }
          
//          pqList.sort {$0.confidence>$1.confidence}
          while (pqList.count>0) {
              let max : Inference = pqList.dequeue() ?? Inference(rect: CGRect.zero,confidence: 0,className: "123")
              nmsList.append(max)
//              var tempList = Heap<Inference>(elements: [], priorityFunction: {
//                  (a, b) in
//              return a.confidence > b.confidence
//              })
              var detections : [Inference] = [];
              while (pqList.count>0) {
                  detections.append(pqList.dequeue() ?? Inference(rect: CGRect.zero,confidence: 0,className: "123") )
              }
              pqList.elements=[]
//                  let detection : Inference = pqList.dequeue() ?? Inference(rect: CGRect.zero,confidence: 0,className: "123")
              
              for  j in (0..<detections.count) {
                  let b : CGRect = detections[j].rect;
                  if (self.box_iou(a: max.rect,b: b) < self.mNmsThresh) {
                      pqList.enqueue(detections[j])
                  }
              }
//            pqList = tempList
          }
      }
      return nmsList;
      
  }
    private func box_iou(a : CGRect, b : CGRect) -> Float{
        return self.box_intersection(a: a, b: b) / self.box_union(a:a, b:b);
        }

    private func box_intersection(a : CGRect, b : CGRect) -> Float {
        let w = self.overlap(x1: Float(a.midX), w1: Float(a.size.width),
                             x2: Float(b.midX), w2: Float(b.size.width));
        let h = self.overlap(x1: Float(a.midY), w1: Float(a.size.height),
                             x2: Float(b.midY), w2: Float(b.size.height));
        
        if (w < 0 || h < 0) {
            return 0;
        }
        let area : Float = w * h;
            return area;
        }

    private func box_union(a : CGRect, b : CGRect) -> Float {
        let i = self.box_intersection(a:a, b:b);
        let u = Float(a.size.width * a.size.height + b.size.width * b.size.height) - i;
            return u;
        }

    private func overlap(x1 : Float, w1 : Float, x2 : Float, w2 : Float) -> Float {
            let l1 = x1 - w1 / 2;
            let l2 = x2 - w2 / 2;
            let left = l1 > l2 ? l1 : l2;
            let r1 = x1 + w1 / 2;
            let r2 = x2 + w2 / 2;
            let right = r1 < r2 ? r1 : r2;
            return right - left;
        }

  /// Loads the labels from the labels file and stores them in the `labels` property.
  private func loadLabels(fileInfo: FileInfo) {
    let filename = fileInfo.name
    let fileExtension = fileInfo.extension
    guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name " +
                   "\(filename).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      labels = contents.components(separatedBy: .newlines)
    } catch {
      fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                   "valid labels file and try again.")
    }
  }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
    ///

  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width
    
    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      print("Error: out of memory")
      return nil
    }
    
    defer {
        free(destinationData)
    }

    var destinationBuffer = vImage_Buffer(data: destinationData,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: destinationBytesPerRow)

    let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)

    switch (pixelBufferFormat) {
    case kCVPixelFormatType_32BGRA:
        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32ARGB:
        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32RGBA:
        vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    default:
        // Unknown pixel format.
        return nil
    }

    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
    if isModelQuantized {
        return byteData
    }

    // Not quantized, convert to floats
    let bytes = Array<UInt8>(unsafeData: byteData)!
    var floats = [Float]()
    for i in 0..<bytes.count {
        floats.append(Float(bytes[i]) / 255.0)
    }
    return Data(copyingBufferOf: floats)
  }
}

// MARK: - Extensions

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}
