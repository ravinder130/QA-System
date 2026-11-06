import AppKit
import Foundation
import PDFKit
import Vision

guard CommandLine.arguments.count == 2 else {
    fputs("Usage: swift ocr_pdf.swift <pdf-path>\n", stderr)
    exit(2)
}

let pdfURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard let document = PDFDocument(url: pdfURL) else {
    fputs("Could not open PDF: \(pdfURL.path)\n", stderr)
    exit(1)
}

for pageNumber in 0..<document.pageCount {
    guard let page = document.page(at: pageNumber) else { continue }
    let thumbnail = page.thumbnail(of: CGSize(width: 2200, height: 2200), for: .mediaBox)
    var proposedRect = CGRect(origin: .zero, size: thumbnail.size)
    guard let image = thumbnail.cgImage(
        forProposedRect: &proposedRect,
        context: nil,
        hints: nil
    ) else { continue }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["en-US"]
    request.usesLanguageCorrection = true

    do {
        try VNImageRequestHandler(cgImage: image).perform([request])
        let observations = (request.results ?? []).sorted {
            if abs($0.boundingBox.midY - $1.boundingBox.midY) > 0.012 {
                return $0.boundingBox.midY > $1.boundingBox.midY
            }
            return $0.boundingBox.minX < $1.boundingBox.minX
        }
        for observation in observations {
            if let candidate = observation.topCandidates(1).first {
                print(candidate.string)
            }
        }
        print("")
    } catch {
        fputs("OCR failed on page \(pageNumber + 1): \(error)\n", stderr)
    }
}