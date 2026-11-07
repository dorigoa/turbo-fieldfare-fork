import Foundation
import Testing
@testable import TurboFieldfare

@Suite struct QuantProfileTests {
    private static func slot(_ bits: Int) -> ManifestQuantSlot {
        ManifestQuantSlot(weightBits: bits, scheme: "affine", scaleType: "bf16",
                          biasType: "bf16", groupSize: Quantization.groupSize)
    }

    private static func quant(_ embedding: Int, _ attention: Int, _ router: Int,
                              _ shared: Int, _ routed: Int) -> ManifestQuant {
        ManifestQuant(embedding: slot(embedding), attention: slot(attention),
                      router: slot(router), sharedExpert: slot(shared),
                      routedExpert: slot(routed))
    }

    @Test func pinnedFourBitCheckpointIsAffine4() throws {
        #expect(try QuantProfile(manifestQuant: Self.quant(4, 4, 8, 4, 4)) == .affine4)
    }

    @Test func historicalEightBitSharedExpertStaysAffine4() throws {
        #expect(try QuantProfile(manifestQuant: Self.quant(4, 4, 8, 8, 4)) == .affine4)
    }

    @Test func pinnedEightBitCheckpointIsAffine8() throws {
        #expect(try QuantProfile(manifestQuant: Self.quant(8, 8, 8, 8, 8)) == .affine8)
    }

    @Test func fourBitRouterIsNotAProfile() {
        #expect(throws: ModelError.self) {
            _ = try QuantProfile(manifestQuant: Self.quant(4, 4, 4, 4, 4))
        }
        #expect(throws: ModelError.self) {
            _ = try QuantProfile(manifestQuant: Self.quant(8, 8, 4, 8, 8))
        }
    }

    @Test func bitAccessorsFollowTheProfile() {
        #expect(QuantProfile.affine4.denseWeightBits == 4)
        #expect(QuantProfile.affine4.routedWeightBits == 4)
        #expect(QuantProfile.affine8.denseWeightBits == 8)
        #expect(QuantProfile.affine8.routedWeightBits == 8)
        #expect(QuantProfile.allCases.allSatisfy { $0.routerWeightBits == 8 })
    }

    /// Until the 8-bit kernels are wired in, an 8-bit model must stop at the
    /// runner's door instead of reaching the 4-bit unpack.
    @Test func runnerAdmitsOnlyAffine4() throws {
        try RealForwardRunner.requireSupportedProfile(.affine4)
        #expect(throws: QuantProfileError.runtimeKernelsUnavailable(.affine8)) {
            try RealForwardRunner.requireSupportedProfile(.affine8)
        }
    }
}

/// The INT8 modules registered for the 8-bit kernels must compile from the
/// start, empty or not, or every Metal test fails at library creation.
@Suite struct Int8ShaderModuleRegistrationTests {
    @Test func sharedLibraryCompilesWithTheInt8Modules() throws {
        _ = try MetalContext()
    }

    @Test func privateInt8TensorOpsModuleCompiles() throws {
        let context = try MetalContext()
        _ = try MetalContext.privateLibrary(device: context.device, module: "tensorops_int8")
    }
}
