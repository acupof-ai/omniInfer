# CUDA IPC relay

The CUDA IPC relay moves packed `StagePayload` host bytes between two stages on
one host. It is not a device-tensor transport and runs no model code.

## Data path

1. The sender packs `StagePayload` into host bytes.
2. `put` takes the device block bound to its credit (allocating or growing it
   on first use), copies the bytes host-to-device, and obtains an IPC handle
   for that block.
3. `DataReady` carries the handle and `transfer_info.size` over the control
   plane.
4. The receiver validates the `DataRef` and opens the handle.
5. `get` copies device-to-host directly from the mapped peer pointer.
6. The receiver acknowledges. The sender releases the credit.

The payload is at most `slot_size_mb` (default 512 MiB) and at most
`max_payload_bytes` (default 1 GiB) on the receive side. A nonpositive
`slot_size_mb` takes the default; a value above `INT_MAX` is a configuration
error raised before any relay or device allocation.

## Operation states

```text
                 put()
                   |
                   v
              [allocated] ---------------------------+
                   |                                 |
        metadata sent, waiting for ack               |
                   |                                 |
      +------------+-------------+                   |
      |            |             |                   |
 mark_done   mark_failed   mark_abandoned        timeout
      |            |             |                   |
      v            v             v                   v
  [completed]  [failed]     [abandoned]     (calls mark_abandoned)
      |            |             |                   |
      +------------+-------------+-------------------+
                   |
             operation destroyed
                   |
     done ack or failure ack -> slot stays in the pool
     abandoned or never acked -> slot leaves the pool, never freed or reused
```

A done ack and a failure ack both mean the receiver has finished with the
mapping, so the slot can be reused. A dropped slot is never freed or reused: a
late receiver may still map it. The CUDA context reclaims that allocation at
process exit.

An abandon, timeout or shutdown marks that target's edge failed: later `put`
calls on it throw until the sender process restarts. A failure ack is not an
edge failure; it settles only the current operation and releases its credit.
Other targets have their own relay and keep working.

Each credit keeps its high-water device allocation until the shared relay
state is destroyed, so steady-state device memory can approach the sum of the
per-credit high-water sizes. An abandoned allocation is retained until process
exit.

## Ownership

- `CommRouter` owns one relay per transport kind, and one CUDA relay per target.
  `close()` closes all of them but keeps them allocated until the router is
  destroyed, because callers may still hold raw relay pointers. After `close()`,
  `relay()` and `relay_for_target()` reject new work instead of building or
  returning a relay.
- Injected relays are test seams and are not owned by the router.
- Each operation holds one credit; the credit owns one device block (its slot).
- A slot is reused only after a done ack or a failure ack. An abandoned slot
  leaves the pool and is never freed or reused, so a late receiver never sees
  it reused.
- The receiver owns the opened mapping for the duration of `get`; the mapping
  is closed before `get` returns. Mappings are not cached.
- Dropped slots belong to the process: their allocation is reclaimed by the
  CUDA context at exit. Pooled slots are freed when the shared relay state is
  destroyed.

## Locking

- `CudaIpcShared::mu` guards `failed`, `failure`, and the slot pool.
- `CreditAllocator::mu` guards the free list and `closed_`; `close()` notifies
  every waiter, and `acquire()` throws after close.
- `CommRouter::relays_mu` guards both relay maps and the `closed_` flag.
  `close()` never clears the maps, so a pointer from `active_relays()` stays
  valid after the lock is dropped.
- `CommRouter::cuda_mu` guards the peer-access cache. The process-wide fallback
  log deduplication set has its own static mutex.
- `Stage::pending_mu` guards the pending operation map.
- Lock order: `pending_mu_` and `relays_mu_` are never held while taking
  `CudaIpcShared::mu`. `CreditAllocator` does not call back into a relay.

## Failure rules

- Ack, failure and abandon are each terminal once; later calls are no-ops.
- A failed ack releases only that operation's credit.
- A timeout or shutdown abandons the block: it is dropped from the pool and
  never freed or reused.
- An exception before `put` returns releases the credit. The slot stays pooled:
  no receiver owns the block, and the next holder of the credit reuses it.
- Zero-length transfers carry a valid handle and clear the destination.
