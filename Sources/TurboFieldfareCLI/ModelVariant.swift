import Foundation
import TurboFieldfare

/// The installed model's variant, as the CLI reports it and admits it.
///
/// The variant is never a flag. A 4-bit and an 8-bit pack share every tensor
/// name and differ only in `manifest.quant`, the one record that cannot
/// disagree with the weights it describes, so the CLI reads it from there.
struct CLIModelVariant: Equatable {
    let profile: QuantProfile
    /// `manifest.modelID`: the Hugging Face repository the pack was built from.
    let manifestModelID: String

    /// "4-bit" or "8-bit", the names the installer and the Mac app use.
    var label: String { "\(profile.denseWeightBits)-bit" }

    /// What admission charges this model beyond its KV cache.
    var footprint: ModelFootprint { .forProfile(profile) }

    /// Reads `manifest.json` alone, a few kilobytes, and never the weights.
    /// The same checks `Model.load` applies, so a manifest that fails here
    /// would have failed the load, only later and after the tokenizer work.
    static func read(modelDirectory: URL,
                     config: ArchConfig = .gemma4_26B_A4B) throws -> CLIModelVariant {
        let manifest = try ManifestReader.load(directoryURL: modelDirectory, expecting: config)
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required to size the model's memory")
        }
        return CLIModelVariant(profile: try QuantProfile(manifestQuant: quant),
                               manifestModelID: manifest.modelID)
    }

    /// The warning printed when this host is below the admission estimate for
    /// the context, or nil. It names the variant because the same context
    /// needs more memory with the 8-bit model, and a reader who knows the
    /// 4-bit figures would otherwise see a number that looks wrong.
    func memoryWarning(maxContext: Int,
                       hostMemoryBytes: UInt64,
                       expertCacheSlots: Int,
                       config: ArchConfig = .gemma4_26B_A4B) -> String? {
        guard case .needsMemory = ContextAdmission.availability(
            config: config,
            maxContext: maxContext,
            hostMemoryBytes: hostMemoryBytes,
            expertCacheSlots: expertCacheSlots,
            footprint: footprint) else { return nil }
        let need = ContextAdmission.needDescription(
            config: config,
            maxContext: maxContext,
            hostMemoryBytes: hostMemoryBytes,
            expertCacheSlots: expertCacheSlots,
            footprint: footprint)
        return "[memory: \(label) model: \(need) Continuing.]"
    }

    /// The variant's footer line. It is a line of its own, printed before the
    /// timing line, so `[stop=...]` keeps its exact shape and stays the last
    /// line for anything that already parses it.
    var footerLine: String {
        "[variant=\(label) manifest_model_id=\(manifestModelID)]"
    }
}

/// The complete footer written to standard error after a run.
func timingFooter(reason: String,
                  prefillTokens: Int,
                  newTokens: Int,
                  decodeSeconds: Double,
                  variant: CLIModelVariant) -> String {
    let tokensPerSecond = decodeSeconds > 0 ? Double(newTokens) / decodeSeconds : 0
    let timing = "[stop=\(reason) prefill=\(prefillTokens)tok new=\(newTokens)tok "
        + "decode=\(String(format: "%.2f", decodeSeconds))s "
        + "tok/s=\(String(format: "%.3f", tokensPerSecond))]"
    return "\n\(variant.footerLine)\n\(timing)\n"
}
