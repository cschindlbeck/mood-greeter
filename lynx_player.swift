// Plays one audio file and tells the caller when the sound starts.
//
// usage: lynx_player <file>
// stdout: "START <seconds>" = the sound begins that many seconds after this line.
// The process exits when playback ends. lynx_voice.py uses the start time for lip sync.
// `afplay` cannot do this: its start delay is not known and it exits up to 0.9 s late.
import AVFoundation
import Foundation

setvbuf(stdout, nil, _IONBF, 0)

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write("usage: lynx_player <file>\n".data(using: .utf8)!)
    exit(2)
}

let url = URL(fileURLWithPath: CommandLine.arguments[1])
guard let player = try? AVAudioPlayer(contentsOf: url), player.prepareToPlay() else {
    FileHandle.standardError.write("lynx_player: cannot play \(url.path)\n".data(using: .utf8)!)
    exit(1)
}

// A short lead gives the audio device time to start, so the start time is exact.
let lead = 0.25
let startAt = player.deviceCurrentTime + lead
guard player.play(atTime: startAt) else { exit(1) }
print("START \(startAt - player.deviceCurrentTime)")

let deadline = Date().addingTimeInterval(lead + player.duration + 3)
while player.isPlaying && Date() < deadline {
    RunLoop.current.run(until: Date().addingTimeInterval(0.02))
}
