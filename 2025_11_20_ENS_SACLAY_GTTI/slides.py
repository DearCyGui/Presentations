
import dearcygui as dcg
from slideshow import SlideShow, Slide, CenterV, CenterH, FootNote, Columns, Column, InteractiveCode, Text, Fill, FillV, FillH


def build_slides(slideshow: SlideShow):
    C = slideshow.context
    with slideshow:
        with Slide("Today's CPUs are fast"):
            with CenterV:
              with FillH(0.8):
                with CenterH(1.):
                    Text("# Today's CPUs are Fast")
                with CenterH(1.):
                    Text("## A Journey from 1982 to 2025")
                with CenterH(1.):
                    Text("")
                with CenterH(1.):
                    Text("**Axel Davy**")
                with CenterH(1.):
                    Text("*27th November 2025*")

        #with Slide("Plan"):
            #Plan()

        #Section("A deep dive into a not so long past...")

        with Slide("The machine"):
            Text("""
## The Machine: Algebra FX (Graph100) Calculator

**Released:** 1999

**Processor:** NEC V30Mx (1994)
- 16-bit processing unit compatible with Intel 80286 and (RM-)DOS
- 5MHz clock speed (equivalent to Intel's 1982 CPUs)
- 80286 assembly instruction set

**Memory:** 64KB of RAM

*A window into the past of computing...*
""")
            with FootNote:
                Text("Reference: `http://cpudb.stanford.edu/processors/349.html`")

        with Slide("Understanding Assembly"):
            with Columns:
                with Column(0.45):
                    with FillV:
                        Text("**Registers:**")
                        gpr_table = dcg.Table(C, header=True, flags=dcg.TableFlag.BORDERS | dcg.TableFlag.SIZING_STRETCH_PROP)
                        gpr_table.col_config[0].label = "Register"
                        gpr_table.col_config[1].label = "High"
                        gpr_table.col_config[2].label = "Low"
                        gpr_table.col_config[3].label = "Special Uses"
                        gpr_table.col_config[3].stretch_weight = 3.
                        
                        gpr_table.append_row(["AW", "AH", "AL", "Integer Ops, I/O"])
                        gpr_table.append_row(["BW", "BH", "BL", "I/O"])
                        gpr_table.append_row(["CW", "CH", "CL", "Loop control, Shifts"])
                        gpr_table.append_row(["DW", "DH", "DL", "addresses"])
                        
                        # Segment Registers
                        Text("**Segment Registers (16-bit):**")
                        seg_table = dcg.Table(C, header=True, flags=dcg.TableFlag.BORDERS | dcg.TableFlag.SIZING_STRETCH_PROP)
                        seg_table.col_config[0].label = "Register"
                        seg_table.col_config[1].label = "Purpose"
                        seg_table.col_config[1].stretch_weight = 3.
                        
                        seg_table.append_row(["PS", "Program segment base"])
                        seg_table.append_row(["SS", "Stack segment base"])
                        seg_table.append_row(["DS0/DS1", "Data segment base"])
                        
                        # Other Registers
                        Text("**Other Registers:**")
                        other_table = dcg.Table(C, header=True, flags=dcg.TableFlag.BORDERS | dcg.TableFlag.SIZING_STRETCH_PROP)
                        other_table.col_config[0].label = "Register"
                        other_table.col_config[1].label = "Purpose"
                        other_table.col_config[1].stretch_weight = 3.
                        
                        other_table.append_row(["SP/BP", "Stack and base pointers"])
                        other_table.append_row(["PC", "Program counter"])
                        other_table.append_row(["IX/IY", "Index registers"])
                        
                        # PSW Flags
                        Text("""
**Program Status Word (PSW) - Flags:**
- **Status**: CY, P, AC, Z, S, V (set by operations)
- **Control**: DIR, IE, BRK, MD (control CPU)
""")
                with Column(0.45):
                    with FillV:
                        Text("**Instructions:**", no_newline=True)
                        Text("""Each with specific encoding, __cycle counts__, and effects on registers and memory.""")
                        inst_table = dcg.Table(C, header=True, flags=dcg.TableFlag.BORDERS | dcg.TableFlag.SIZING_STRETCH_PROP)
                        inst_table.col_config[0].label = "Instruction Group"
                        inst_table.col_config[1].label = "Examples"
                        inst_table.col_config[1].stretch_weight = 1.5

                        inst_table.append_row(["Data transfer", "MOV, XCH, TRANS"])
                        inst_table.append_row(["Repeat prefix", "REP, REPE, REPNE"])
                        inst_table.append_row(["Block transfer", "MOVBK, CMPBK, LDM, STM"])
                        inst_table.append_row(["Bit field", "EXT, INS"])
                        inst_table.append_row(["I/O", "IN, OUT, INM, OUTM"])
                        inst_table.append_row(["Add/subtract", "ADD, ADDC, SUB, SUBC"])
                        inst_table.append_row(["BCD operations", "ADD4S, SUB4S, ROL4, ROR4"])
                        inst_table.append_row(["Inc/decrement", "INC, DEC"])
                        inst_table.append_row(["Multiply/divide", "MUL, MULU, DIV, DIVU"])
                        inst_table.append_row(["BCD adjust", "ADJ4A, ADJ4S, ADJBA"])
                        inst_table.append_row(["Data convert", "CVTBD, CVTBW, CVTDB"])
                        inst_table.append_row(["Compare", "CMP"])
                        inst_table.append_row(["Complement", "NEG, NOT"])
                        inst_table.append_row(["Logical", "AND, OR, XOR, TEST"])
                        inst_table.append_row(["Bit manipulation", "CLR1, SET1, NOT1, TEST1"])
                        inst_table.append_row(["Shift", "SHL, SHR, SHRA"])
                        inst_table.append_row(["Rotate", "ROL, ROR, ROLC, RORC"])
                        inst_table.append_row(["Subroutine", "CALL, RET"])
                        inst_table.append_row(["Stack", "PUSH, POP, PREPARE"])
                        inst_table.append_row(["Branch", "BR, JMP"])
                        inst_table.append_row(["Conditional branch", "BE, BNE, BL, BGE, BZ, ..."])
                        inst_table.append_row(["Interrupt", "BRK, BRKV, RETI"])
                        inst_table.append_row(["CPU control", "HALT, NOP, DI, EI, POLL"])
                        inst_table.append_row(["Segment override", "DS0:, DS1:, PS:, SS:"])
                    

        with Slide("The assembly"):
            Text("Understanding RISC vs CISC: Implementing a screen shift in assembly")
            Text("""
The operation: **Vertical shift up by one row** of a `128x64` pixel screen buffer.

Each pixel is 3-bit grayscale (8 shades), requiring **3 screen buffers** (one bit per buffer per pixel).
Each screen buffer is `128x64` bits, organized as 64 rows of 16 bytes (128 bits = 16 bytes).
Total memory: 3 buffers x 64 rows x 16 bytes = **3,072 bytes**

*Visualization shows one buffer; operation is repeated for all 3 buffers.*
""")
            
            with Columns:
                with Column(0.48):
                    with CenterH:
                        Text("### BEFORE")
                    with dcg.DrawInWindow(C, width="fillx", height=400, relative=True):
                        # Draw screen buffer representation
                        buffer_x = 0.1
                        buffer_width = 0.8
                        buffer_y = 0.05
                        buffer_height = 0.85

                        # background box
                        dcg.DrawRect(C, 
                                     pmin=(buffer_x,
                                           buffer_y), 
                                     pmax=(buffer_x + buffer_width,
                                           buffer_y + buffer_height),
                                     color=0,
                                     fill=(30, 30, 40),
                                     thickness=0.)

                        # Draw some rows with pattern to show data
                        num_rows_shown = 64
                        for row in range(num_rows_shown):
                            y = buffer_y + (row / num_rows_shown) * buffer_height
                            row_h = buffer_height / num_rows_shown
                            
                            # Create a visual pattern
                            alpha = 200 if row % 4 == 0 else 120
                            color = (100, 150, 255, alpha)
                            dcg.DrawLine(C,
                                            p1=(buffer_x + 0.0005,
                                                y + 0.0005), 
                                            p2=(buffer_x + buffer_width - 0.0005,
                                                y + 0.0005),
                                            color=color,
                                            thickness=0.001)
                        
                        # Screen outline
                        dcg.DrawRect(C, 
                                     pmin=(buffer_x + 0.001,
                                           buffer_y + 0.001), 
                                     pmax=(buffer_x + buffer_width - 0.001,
                                           buffer_y + buffer_height - 0.001),
                                     color=(150, 150, 150),
                                     fill=0,
                                     thickness=0.002)

                        # Labels for key rows
                        dcg.DrawText(C, pos=(0.02, buffer_y), 
                                   text="Row 0", color=(200, 200, 200), size=-12)
                        dcg.DrawText(C, pos=(0.02, buffer_y + buffer_height / num_rows_shown), 
                                   text="Row 1", color=(200, 200, 200), size=-12)
                        dcg.DrawText(C, pos=(0.02, buffer_y + buffer_height - 0.02), 
                                   text="Row 63", color=(200, 200, 200), size=-12)
                        
                        # Highlight the first row that will be lost
                        dcg.DrawLine(C,
                                     p1=(buffer_x + 0.001, buffer_y), 
                                     p2=(buffer_x + buffer_width - 0.001, buffer_y),
                                     color=(255, 80, 80),
                                     thickness=0.002)
                        dcg.DrawText(C, pos=(buffer_x + buffer_width + 0.02, buffer_y), 
                                   text="Discarded", color=(255, 100, 100), size=-12)
                
                with Column(0.48):
                    with CenterH:
                        Text("### AFTER")
                    with dcg.DrawInWindow(C, width=-1, height=400, relative=True):
                        # Draw screen buffer representation
                        buffer_x = 0.1
                        buffer_width = 0.8
                        buffer_y = 0.05
                        buffer_height = 0.85

                        # background box
                        dcg.DrawRect(C, 
                                     pmin=(buffer_x,
                                           buffer_y), 
                                     pmax=(buffer_x + buffer_width,
                                           buffer_y + buffer_height),
                                     color=0,
                                     fill=(30, 30, 40),
                                     thickness=0.)

                        # Draw shifted rows
                        num_rows_shown = 64
                        for row in range(num_rows_shown):
                            y = buffer_y + (row / num_rows_shown) * buffer_height
                            row_h = buffer_height / num_rows_shown
                            
                            # Create a visual pattern
                            alpha = 200 if row % 4 == 0 else 120
                            color = (100, 150, 255, alpha)
                            dcg.DrawLine(C,
                                            p1=(buffer_x + 0.0005,
                                                y + 0.0005), 
                                            p2=(buffer_x + buffer_width - 0.0005,
                                                y + 0.0005),
                                            color=color,
                                            thickness=0.001)
                        
                        # Screen outline
                        dcg.DrawRect(C, 
                                     pmin=(buffer_x + 0.001,
                                           buffer_y + 0.001), 
                                     pmax=(buffer_x + buffer_width - 0.001,
                                           buffer_y + buffer_height - 0.001),
                                     color=(150, 150, 150),
                                     fill=0,
                                     thickness=0.002)

                        # Labels for key rows
                        row_h = buffer_height / num_rows_shown
                        dcg.DrawText(C, pos=(buffer_x + buffer_width + 0.02, buffer_y), 
                                   text="Row 1 data", color=(200, 200, 200), size=-11)
                        dcg.DrawText(C, pos=(buffer_x + buffer_width + 0.02, buffer_y + row_h), 
                                   text="Row 2 data", color=(200, 200, 200), size=-11)
                        
                        # Highlight the last row (new data from source at -y offset)
                        last_y = buffer_y + buffer_height - row_h
                        dcg.DrawLine(C,
                                     p1=(buffer_x + 0.001, buffer_y + buffer_height - 0.001), 
                                     p2=(buffer_x + buffer_width - 0.001, buffer_y + buffer_height - 0.001),
                                     color=(100, 255, 100),
                                     thickness=0.002)
                        dcg.DrawText(C, pos=(0.02, last_y), 
                                   text="New data", color=(100, 255, 100), size=-12)
            
            Text("")
            Text("Each row: 128 bits = 16 bytes  |  Total per buffer: 64 rows x 16 bytes = 1,024 bytes")
            Text("Complete operation: 3 buffers x 1,024 bytes = 3,072 bytes to process")

        with Slide("Assembly Implementation: Naive Approach"):
            Text("""
## Screen Shift Implementation: Variant 1 (Naive)

This variant uses simple MOV instructions to copy memory word by word.

**Performance:** 58,504 cycles
""")
            with Columns:
                with Column:
                    Text("""
### Initialization (4 cycles)
```asm
mov al, x           ; Load repeat count
mov dl, al          ; Save copy
```

### Iteration setup (26 cycles)
```asm
nextbuffer:
mov ax, segm        ; Set buffer segment
mov es, ax
mov di, 0x10        ; di = 16 (data block)
mov si, 0
sub si, 0x3F        ; si = -y
shl si, 0x4         ; si = -y * 16
add si, 0x3F0       ; si points to source
mov cx, 0x1F8       ; Loop 504 times (8*63)
```
""")
                with Column:
                    Text("""
### Main Copy Loop (19,114 cycles)
```asm
loopline:           ; 504 iterations
mov ax, es:[di]     ; Read word at di
mov es:[di-0x10], ax ; Copy to line above
add di, 0x2         ; Next word
LOOP loopline       ; Repeat cx times

or ch, 0x8          ; Reset ch
sub di, 0x10        ; Adjust pointer
```

### Bottom Line Fill (143 cycles)
```asm
loop:
mov ax, es:[si]     ; Read from source line
mov es:[di], ax     ; Write to buffer
add di, 0x2         ; Next dest word
add si, 0x2         ; Next source word
dec ch              ; Decrement counter
jne loop            ; Continue if not zero
dec dl              ; Decrement buffer count
jne nextbuffer      ; Repeat for all buffers
```
""")

        with Slide("Assembly Implementation: Optimized Approach"):
            Text("""
## Screen Shift Implementation: Variant 2 (Optimized)

This variant uses **CISC-specific** `REP MOVSW` instruction for block memory copies.

**Performance:** ~12,550 cycles (**0.0025 seconds** at 5MHz)

**Speedup:** 4.6x faster!
""")
            with Columns:
                with Column:
                    Text("""
### Setup (15 cycles)
```asm
push ds
mov ax, 0x1A60      ; Set buffer 1
mov es, ax
mov ds, ax
mov dh, 3           ; 3 buffers to process
cld                 ; Clear direction flag
```

### Per-Buffer Init (6 cycles)
```asm
nextbuffer:
mov al, x           ; Load repeat count
mov dl, al
```

### Main Shift (4,049 cycles)
```asm
retu:
mov di, 0x0         ; Destination offset
mov si, 0x10        ; Source offset
mov cx, 0x1F8       ; 504 words
rep movsw           ; Copy entire block!
```
""")
                with Column:
                    Text("""
### Bottom Line Fill (107 cycles)
```asm
mov cx, 0x8         ; 8 words
mov si, 0
sub si, 0x3F        ; si = -y
shl si, 0x4         ; si = -y * 16
add si, 0x3F0       ; Point to source
rep movsw           ; Copy bottom line!

dec dl              ; Decrement repeat
jne retu            ; Continue if needed
```

### Next Buffer (14 cycles, final: 6)
```asm
mov ax, es
add ax, 0x80        ; Next buffer segment
mov es, ax
mov ds, ax
dec dh              ; Decrement buffer count
jne nextbuffer      ; Process next buffer
pop ds              ; Restore ds
```

**Key Insight:** `REP MOVSW` is a single CISC instruction that performs what would take many RISC instructions!
""")

        with Slide("Compiled vs Interpreted"):
            with Fill:
                Text("""
## Compiled vs Interpreted Languages

### Instruction Encoding
An instruction contains a packed binary representation:
- **The operation to perform**  
Examples: Add, Jump-if-zero, Compare, Read-from-memory
- **The input(s) and output**  
Examples: Register index, memory offset to an address

### Key Differences

**Compiled (e.g., C, Rust):**
- Assembly is executed directly by the CPU
- No runtime checks needed
- Instructions define types of inputs at compile time

**Interpreted (e.g., Python, Lua):**
- Text/bytecode converted to operations at runtime
- Runtime type checks required
- Inputs have intrinsic types that must be verified
- Opcodes parsed, extracted, then corresponding function performed
""")

        with Slide("Performance Impact"):
            Text("## The Performance Gap")
            with Columns:
                with Column(0.45):
                    with FillV:
                        Text("""
### Speed Difference

**Interpreted overhead includes:**
- Opcode parsing
- Type checking at runtime
- Dynamic dispatch/jumps
- Memory allocations

**Historical example:**
- AFX Lua (1999): **50x slower** than compiled code for integer addition

**Modern Python:**
- Still significant overhead
- Let's measure it
""")
                with Column(0.45):
                    with Fill:
                        interactive = InteractiveCode(C, """
import numpy as np
from timeit import timeit
def add_python():
    a = 5
    b = 6
    c = a + b
    c = a + b
    c = a + b
    c = a + b
    c = a + b
    c = a + b
    c = a + b
    c = a + b
    c = a + b
    c = a + b
arr = np.arange(1000_000)
t1 = timeit(add_python, number=100_000_000)
t2 = timeit(lambda: arr.sum(), number=1000)
print(f"Interpreted: {t1:.4f}s, Compiled: {t2:.4f}s (Ratio {t1/t2:.1f}x)")
""")
                        interactive.display_result()

        with Slide("CPU Evolution"):
            with FillV(0.9):
                Text("""
## How Modern CPUs Achieve Better Performance

### Key Innovations Over Time

**1. Higher Clock Frequencies**
- From 5MHz (1982) to 5GHz+ (2025) - 1000x increase

**2. Reduced Cycles Per Instruction (CPI)**
- Better microarchitecture implementations
- Smarter execution pipelines

**3. Pipelining (~1986)**
- Overlapping execution of instructions
- Multiple stages work simultaneously

**4. Superscalar Execution**
- Decode multiple instructions in large buffers
- Convert to micro-ops (µops)
- Out-of-order execution with dependency analysis
- CISC penalty: ~4 IPC (Intel/AMD) vs ~8 IPC (Apple M1)

**5. Extended Instruction Sets**
- SIMD instructions for parallel data processing
- Multiplicative performance gains for specific workloads

**6. Memory Hierarchy & Multi-core**
- Larger, smarter caches
- Multiple cores for parallel execution
""")
            with FootNote:
                Text("Reference: `https://arxiv.org/pdf/1803.00254`")

        with Slide("Floating Point Revolution"):
            with FillV(0.7):
                Text("""
## The Floating Point Revolution

### Then: NEC V30Mx
- **No native FP support**
- Software emulation required
- Float addition: ~300 clock cycles
- Let's not even discuss multiplication...

### Now: Modern CPUs
- FP operations **faster than integers** (with optimization)
- CISC solution: **SIMD Instructions**

### SIMD: Single Instruction, Multiple Data
- **SSE/AVX/AVX512**: One instruction, many operations
- Modern CPUs: 4 AVX-512 instructions per clock, per core
- Each AVX-512 instruction: **16 float operations**

### Mind-Blowing Example: AMD Zen 5
**Specs:** 5GHz, 8 cores

**FP32 Performance:**
- 512 FMA + 512 FADD operations **per clock cycle**
- Total: **5 TFlops**
- *More than most integrated GPUs from just a few years ago*

**FP64 Performance:**
- 2.5 TFlops
- **4x the performance of an NVIDIA RTX 3090 Ti GPU!**
""")
            with FootNote:
                Text("RTX 3090 Ti: `https://www.techpowerup.com/gpu-specs/geforce-rtx-3090-ti.c3829`")
                Text("Discussion: `https://news.ycombinator.com/item?id=41438343`")

        with Slide("Modern Bottlenecks"):
            with FillV:
                Text("""
## What's Limiting CPUs Now?

### 1. Memory Bandwidth & Latency
With such fast computation, **retrieving instructions and data** becomes the primary bottleneck.

### 2. Code Complexity vs Simplicity
- **RISC CPUs** can decode instructions more cheaply
- Simpler instruction streams = more throughput
- Modern ARM/RISC-V gaining ground

### 3. Data-Oriented vs Object-Oriented Design
As data processing speeds increased, **code logic overhead** became more significant.

**Modern trend:** Data-oriented designs
- C++, Rust moving away from object-based patterns
- Focus on cache-friendly, contiguous data
- Minimizing pointer chasing and branching

### 4. The Interpreted Language Problem
- Interpreted code has **inherent, unshrinkable overhead**
- Type checks
- Dynamic dispatch
- Object allocation
- As CPUs get faster, **the performance gap widens**
- Python, JavaScript increasingly bottlenecked by interpreter logic
""")

        with Slide("Conclusion"):
            with CenterV:
                with FillV:
                    Text("""
## The Incredible Journey

### From 1982 to 2025 (43 years)

**Performance Growth:**
- From 17 KFlops to 5 TFlops
- **294,000x improvement**
- Average: **2x every 2.3 years**

### Key Takeaway

**Your CPU is an extreme powerhouse.**

What took minutes in 1982 now takes microseconds.
What was impossible then is trivial today.

*The computer in your pocket is more powerful than supercomputers of the past.*
""")

if __name__ == "__main__":
    C = dcg.Context()
    slideshow = SlideShow(C, title="Today's CPUs are fast")
    slideshow.scaling_factor = 1.6
    build_slides(slideshow)
    C.viewport.initialize(title="GTTI: Today's CPUs are fast", vsync=True, wait_for_input=True)
    slideshow.start()
    while C.running:
        C.viewport.render_frame()