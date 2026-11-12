---- MODULE RequestLifecycle ----
EXTENDS Naturals

CONSTANT MaxOutputs

VARIABLES state, outputs, started, abortAccepted
vars == <<state, outputs, started, abortAccepted>>

TerminalStates == {"completed", "failed", "aborted"}

TypeOK ==
    /\ state \in {"idle", "submitted", "running"} \cup TerminalStates
    /\ outputs \in 0..MaxOutputs
    /\ started \in BOOLEAN
    /\ abortAccepted \in BOOLEAN

Init ==
    /\ state = "idle"
    /\ outputs = 0
    /\ started = FALSE
    /\ abortAccepted = FALSE

Submit ==
    /\ state = "idle"
    /\ state' = "submitted"
    /\ UNCHANGED <<outputs, started, abortAccepted>>

Start ==
    /\ state = "submitted"
    /\ state' = "running"
    /\ started' = TRUE
    /\ UNCHANGED <<outputs, abortAccepted>>

Emit ==
    /\ state = "running"
    /\ outputs < MaxOutputs
    /\ outputs' = outputs + 1
    /\ UNCHANGED <<state, started, abortAccepted>>

Complete ==
    /\ state = "running"
    /\ state' = "completed"
    /\ UNCHANGED <<outputs, started, abortAccepted>>

Fail ==
    /\ state = "running"
    /\ state' = "failed"
    /\ UNCHANGED <<outputs, started, abortAccepted>>

Abort ==
    /\ state \in {"submitted", "running"}
    /\ state' = "aborted"
    /\ abortAccepted' = TRUE
    /\ UNCHANGED <<outputs, started>>

\* Atomic actions place abort and completion only between token outputs.
\* Terminal stutter avoids deadlock without emitting or changing the result.
TerminalStutter == state \in TerminalStates /\ UNCHANGED vars

Next == Submit \/ Start \/ Emit \/ Complete \/ Fail \/ Abort \/ TerminalStutter
Spec == Init /\ [][Next]_vars

NoOutputBeforeStart == outputs > 0 => started
BoundedOutput == outputs <= MaxOutputs
NoSuccessAfterAcceptedAbort == state = "completed" => ~abortAccepted
TerminalStep == state \in TerminalStates => UNCHANGED vars
TerminalStatesCannotEmitOrTransition == [][TerminalStep]_vars

====
