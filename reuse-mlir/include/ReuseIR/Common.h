#pragma once

#if defined(__GNUG__) && __has_attribute(visibility)
#define REUSE_IR_DECL_SCOPE [[gnu::visibility("hidden")]] reuse_ir
#else
#define REUSE_IR_DECL_SCOPE reuse_ir
#endif
