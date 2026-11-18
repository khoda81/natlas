// A transcript as a standard MIDI file: a track per instrument, timed in
// seconds of the recording, so it lines up with the audio it transcribes.

import { DRUM_PROGRAM } from "./music";
import type { Transcript } from "./notes";

/** Ticks per quarter note, at the default 120 beats a minute: 960 a second. */
const PPQ = 480;
const TICKS_PER_SECOND = 2 * PPQ;
const DRUM_CHANNEL = 9;
/** MuScriptor writes when a drum strikes, not how long it rings. */
const HIT_SECONDS = 0.1;
/** MuScriptor writes notes on or off, not how loud. */
const VELOCITY = 100;

interface Event {
  tick: number;
  bytes: number[];
}

/**
 * ``transcript`` as a format 1 MIDI file. ``seconds`` places a shift step in
 * the recording; ``name`` names an instrument. A note still sounding ends
 * where the transcript does.
 */
export function midi(transcript: Transcript, seconds: (tick: number) => number, name: (program: number) => string): Uint8Array<ArrayBuffer> {
  const at = (tick: number) => Math.round(seconds(tick) * TICKS_PER_SECOND);
  const tracks = new Map<number, Event[]>();
  const track = (program: number) => {
    let events = tracks.get(program);
    if (events === undefined) tracks.set(program, events = []);
    return events;
  };
  // Instruments take the channels in turn, the drums' aside. MIDI has 16, so
  // past 15 instruments some share one, and the last program change wins.
  const channels = new Map<number, number>();
  const channel = (program: number) => {
    if (program === DRUM_PROGRAM) return DRUM_CHANNEL;
    let channel = channels.get(program);
    if (channel === undefined) {
      const index = channels.size % 15;
      channels.set(program, channel = index < DRUM_CHANNEL ? index : index + 1);
    }
    return channel;
  };
  const note = (program: number, pitch: number, from: number, to: number) => {
    const events = track(program);
    const c = channel(program);
    events.push({ tick: from, bytes: [0x90 | c, pitch, VELOCITY] }, { tick: Math.max(from + 1, to), bytes: [0x80 | c, pitch, 0] });
  };
  for (const { program, pitch, start, end } of transcript.notes) note(program, pitch, at(start), at(end ?? transcript.now));
  for (const { pitch, tick } of transcript.hits) {
    const from = at(tick);
    note(DRUM_PROGRAM, pitch, from, from + Math.round(HIT_SECONDS * TICKS_PER_SECOND));
  }

  const chunks = [chunk("MTrk", [...meta(0x51, [0x07, 0xa1, 0x20]), ...meta(0x2f, [])])];
  for (const [program, events] of tracks) {
    const head: number[] = [...meta(0x03, [...new TextEncoder().encode(name(program))])];
    // GM programs are 0 to 127; the drums are a channel, not a program.
    if (program < 128) head.push(0, 0xc0 | channel(program), program);
    // A note ends before another starts at the same tick, so repeated notes sound.
    events.sort((a, b) => a.tick - b.tick || (a.bytes[0]! & 0x10) - (b.bytes[0]! & 0x10));
    let previous = 0;
    const body = events.flatMap(({ tick, bytes }) => {
      const delta = tick - previous;
      previous = tick;
      return [...quantity(delta), ...bytes];
    });
    chunks.push(chunk("MTrk", [...head, ...body, ...meta(0x2f, [])]));
  }
  const header = chunk("MThd", [0, 1, ...u16(chunks.length), ...u16(PPQ)]);
  return Uint8Array.from([...header, ...chunks.flat()]);
}

/** A meta event at no delta. */
function meta(type: number, data: number[]): number[] {
  return [0, 0xff, type, ...quantity(data.length), ...data];
}

function chunk(type: string, data: number[]): number[] {
  return [...type].map((c) => c.charCodeAt(0)).concat(u32(data.length), data);
}

function u16(value: number): number[] {
  return [value >> 8 & 0xff, value & 0xff];
}

function u32(value: number): number[] {
  return [value >>> 24 & 0xff, value >> 16 & 0xff, value >> 8 & 0xff, value & 0xff];
}

/** A variable-length quantity: seven bits a byte, most significant first. */
function quantity(value: number): number[] {
  const bytes = [value & 0x7f];
  for (value >>>= 7; 0 < value; value >>>= 7) bytes.unshift(0x80 | value & 0x7f);
  return bytes;
}
