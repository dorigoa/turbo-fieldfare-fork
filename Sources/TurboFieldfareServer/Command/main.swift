import Darwin
import Foundation
import TurboFieldfare
import TurboFieldfareServerCore

// The ready line below goes to stdout, which is fully buffered when it is not
// a terminal. A server started with its output redirected therefore printed
// "ready" only as it exited - exactly inverted from what an operator needs.
// Line buffering puts the line where it is useful. The request log is written
// line by line without a buffer, to stdout unless `--log-output stderr`.
setvbuf(stdout, nil, _IOLBF, 0)

let arguments: ServerArguments
let runtimeConfiguration: RuntimeConfiguration
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
    // Resolved here so an unusable flag combination exits with usage instead of
    // failing after the model has started loading.
    runtimeConfiguration = try arguments.resolvedRuntimeConfiguration()
    ServerLog.setOutput(arguments.logOutput)
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        visionPackURL: arguments.visionPack.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        },
        visionResidencyPolicy: arguments.visionResidency,
        promptCacheMode: arguments.promptCacheMode,
        runtimeConfiguration: runtimeConfiguration)
    let metrics = ServerMetrics(
        modelLabels: ServerMetrics.modelLabels(arguments: arguments,
                                               variant: backend.modelVariant,
                                               visionCapability: backend.visionCapability),
        maxContextTokens: arguments.maxContext)
    let server = TurboFieldfareHTTPServer(
        modelID: arguments.modelID,
        queueLimit: arguments.queueLimit,
        backend: backend,
        visionCapability: backend.visionCapability,
        family: backend.family,
        metrics: metrics,
        progressLogInterval: arguments.progressLogInterval > 0
            ? .seconds(Int64(arguments.progressLogInterval)) : nil,
        acceptAnyModel: arguments.acceptAnyModel)
    if let warning = arguments.exposureWarning {
        FileHandle.standardError.write(Data("\(warning)\n".utf8))
    }
    _ = try await server.start(host: arguments.host, port: arguments.port)
    print(arguments.readyLine(servedVariant: backend.modelVariant,
                              visionCapability: backend.visionCapability))

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
