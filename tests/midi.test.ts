import { expect, it } from "bun:test";
import { midi } from "../src/readings/midi";
import { DRUM_PROGRAM } from "../src/readings/music";
import type { Transcript } from "../src/readings/notes";

/** Each track's name and channel events, at absolute ticks. */
function read(bytes: Uint8Array): { format: number; ppq: number; tracks: { name: string; events: [number, ...number[]][] }[] } {
  const view = new DataView(bytes.buffer);
  const text = (from: number, length: number) => new TextDecoder().decode(bytes.subarray(from, from + length));
  expect(text(0, 4)).toBe("MThd");
  const tracks = [];
  let at = 14;
  while (at < bytes.length) {
    expect(text(at, 4)).toBe("MTrk");
    const end = at + 8 + view.getUint32(at + 4);
    at += 8;
    let tick = 0;
    let name = "";
    const events: [number, ...number[]][] = [];
    const quantity = () => {
      let value = 0;
      for (let byte = 0x80; byte & 0x80; ) value = (value << 7) | ((byte = bytes[at++]!) & 0x7f);
      return value;
    };
    while (at < end) {
      tick += quantity();
      const status = bytes[at++]!;
      if (status === 0xff) {
        const type = bytes[at++]!;
        const length = quantity();
        if (type === 0x03) name = text(at, length);
        at += length;
      } else {
        const size = (status & 0xf0) === 0xc0 ? 1 : 2;
        events.push([tick, status, ...bytes.subarray(at, at + size)]);
        at += size;
      }
    }
    tracks.push({ name, events });
  }
  return { format: view.getUint16(8), ppq: view.getUint16(12), tracks };
}

it("writes a track per instrument, timed in seconds of the recording", () => {
  const transcript = {
    notes: [
      { program: 24, pitch: 52, start: 0, end: 5, carried: true },
      { program: 24, pitch: 52, start: 5, end: null, carried: false },
    ],
    hits: [{ pitch: 36, tick: 10 }],
    now: 50,
  } as unknown as Transcript;
  // Steps of 10 ms from 1 s into the recording.
  const file = read(midi(transcript, (tick) => 1 + tick / 100, (program) => (program === DRUM_PROGRAM ? "drums" : `#${program}`)));
  expect(file.format).toBe(1);
  // 120 beats a minute: 960 ticks a second.
  expect(file.ppq).toBe(480);
  expect(file.tracks.map(({ name }) => name)).toEqual(["", "#24", "drums"]);
  expect(file.tracks[1]!.events).toEqual([
    [0, 0xc0, 24],
    [960, 0x90, 52, 100],
    // The repeated note ends before it starts again; the open one ends with the transcript.
    [1008, 0x80, 52, 0],
    [1008, 0x90, 52, 100],
    [1440, 0x80, 52, 0],
  ]);
  expect(file.tracks[2]!.events).toEqual([
    [1056, 0x99, 36, 100],
    [1152, 0x89, 36, 0],
  ]);
});
