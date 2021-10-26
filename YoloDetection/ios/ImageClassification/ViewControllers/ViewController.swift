import AVFoundation
import UIKit

class ViewController: UIViewController {

  // MARK: Storyboards Connections
//    @IBOutlet weak var previewView2: PreviewView!
    @IBOutlet weak var countLabel: UILabel!
    @IBOutlet weak var playButton: UIButton!
    // MARK: Instance Variables
  // Holds the results at any time
  private var result: Result?
    private var buff: CVPixelBuffer?

    @IBOutlet weak var imageView: UIImageView!
    // MARK: Controllers that manage functionality
  // Handles all data preprocessing and makes calls to run inference through the `Interpreter`.
  private var modelDataHandler: ModelDataHandler? =
    ModelDataHandler(modelFileInfo: MobileNet.modelInfo, labelsFileInfo: MobileNet.labelsInfo, threadCount: 4)

  // Handles the presenting of results on the screen

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()
      playButton.addTarget(self, action:#selector(self.buttonClicked), for: .touchUpInside)
      self.view.addSubview(playButton)
      self.view.addSubview(imageView)
      setImage()
      
    guard modelDataHandler != nil else {
      fatalError("Model set up failed")
    }
  }
    @objc func buttonClicked() {
        
        DispatchQueue.global().async {
            self.recognizeImage(pixelBuffer: self.buff!)
        }
    }
    
    @objc func recognizeImage(pixelBuffer: CVPixelBuffer) {
        
        // Pass the pixel buffer to TensorFlow Lite to perform inference.
        result = modelDataHandler?.runModel(onFrame: pixelBuffer)
        guard let inferences = result?.inferences else {
              return
            }
          DispatchQueue.main.async {
              let imageDisplay = CIImage(image: self.imageView.image!)
              
              for inference in inferences {
                  var bounds = inference.rect
                  let imageViewSize = self.imageView.bounds.size
                         let scale = min(imageViewSize.width / imageDisplay!.extent.width,
                                         imageViewSize.height / imageDisplay!.extent.height)
                         
                         let dx = (imageViewSize.width - imageDisplay!.extent.width * scale) / 2
                         let dy = (imageViewSize.height - imageDisplay!.extent.height * scale) / 2
                  bounds.applying(CGAffineTransform(scaleX: scale, y: scale))
                  bounds.origin.x += dx
                  bounds.origin.y += dy
                  let box = UIView(frame: bounds)
                      box.layer.borderColor = UIColor.red.cgColor
                      box.layer.borderWidth = 2
                      box.backgroundColor = UIColor.clear
                  self.imageView.addSubview(box)
                  }
              print(self.result?.inferenceTime ?? 0)
              self.countLabel.text="Inference time: " +  String(format: "%.2fms", self.result?.inferenceTime ?? 0.02)
          }
      }

  @objc func setImage() {
    guard let image = UIImage(named: "dog") else {
      return
    }

    guard let buffer = CVImageBuffer.buffer(from: image) else {
      return
    }
      self.buff = buffer
    imageView.image = image
  }

}
