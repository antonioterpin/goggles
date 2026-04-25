# Shutdown drained `BYE` before its own send queue (April 2026)

## What went wrong

A run that emitted ~1500 video-log events before calling
`gg.finish()` consistently delivered only **77 of them** to the host.
The losses were silent — no exceptions, no warnings — and were only
caught by manually counting frames on the receiver.

## Root cause

`LocalTransport` shutdown sent the `BYE` control frame **out-of-band**
relative to the regular send queue:

1. The producer thread was still appending event frames to
   `_send_queue`.
2. `_shutdown_client()` acquired `_send_lock` and `sendall()`'d a
   `BYE` directly.
3. The send loop, observing the `BYE` had been delivered, returned.
4. Anything still sitting in `_send_queue` (typically the tail of the
   producer's batch) was discarded when the loop tore down.

The contract everyone *thought* was being honoured was "shutdown
flushes the queue, then sends BYE". The implementation was "shutdown
sends BYE, then tears down the queue". Under any non-trivial producer
load the latter loses tail events.

## Fix

`_shutdown_client` now enqueues the `BYE` frame as the **last**
element of `_send_queue` and waits for the send loop to drain naturally
before joining the thread. PR
[#143](https://github.com/antonioterpin/goggles/pull/143) lands this
along with the broader Unix-socket transport rewrite.

The regression is guarded by
[`tests/core/test_transport.py::test_shutdown_flushes_pending_events`](../../tests/core/test_transport.py),
which emits 500 events back-to-back and asserts the receiver sees all
500 before `shutdown()` returns.

## Lesson

When a control frame ("end of stream", "drain me", "ack") shares the
same wire as data frames, treat it as a queue-ordered event, not as a
side channel. Out-of-band shortcuts to the socket bypass any flush
discipline the data path is enforcing. If a side channel is genuinely
needed (e.g. a high-priority "abort" signal), it deserves its own
guarantees and its own test for tail losses on the data side.
