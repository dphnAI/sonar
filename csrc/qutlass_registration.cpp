#include "core/registration.h"

// QuTLASS registers torch.ops._qutlass_C via STABLE_TORCH_LIBRARY in
// bindings.cpp. Importing sonar._qutlass_C triggers those stable-ABI
// registrations.
REGISTER_EXTENSION(_qutlass_C)
