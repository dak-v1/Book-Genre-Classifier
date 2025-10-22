//
//  ContentView.swift
//  BookGenreClassifier
//
//  Created by Dakshaa on 22/10/25.
//

import SwiftUI
import AVFoundation
import Vision
import CoreML


final class CameraViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var predictionLabel = "Detecting..."
    public let session = AVCaptureSession()   // made public so SwiftUI view can access it
    private var request: VNCoreMLRequest!

    override init() {
        super.init()
        setupModel()
        setupCamera()
    }

    // Load Core ML model
    func setupModel() {
        guard let coreMLModel = try? StorybookImageClassifier(configuration: MLModelConfiguration()).model,
              let vnModel = try? VNCoreMLModel(for: coreMLModel) else {
            print("Failed to load ML model.")
            return
        }

        request = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
            if let result = req.results?.first as? VNClassificationObservation {
                DispatchQueue.main.async {
                    self?.predictionLabel = "\(result.identifier) \(Int(result.confidence * 100))%"
                }
            }
        }
    }

    // Setup camera session
    func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .photo

        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("Could not access camera.")
            return
        }

        if session.canAddInput(input) { session.addInput(input) }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))

        if session.canAddOutput(output) { session.addOutput(output) }

        session.commitConfiguration()
        session.startRunning()
    }

    // Process each camera frame
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}


struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = UIScreen.main.bounds
        view.layer.addSublayer(previewLayer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) { }
}

// MARK: - ContentView
struct ContentView: View {
    @StateObject private var vm = CameraViewModel()

    var body: some View {
        ZStack(alignment: .bottom) {
            // Live camera feed
            CameraPreview(session: vm.session)
                .ignoresSafeArea()

            // Prediction overlay
            Text(vm.predictionLabel)
                .font(.title)
                .bold()
                .foregroundColor(.white)
                .padding()
                .background(.ultraThinMaterial)
                .cornerRadius(12)
                .padding(.bottom, 40)
        }
    }
}


