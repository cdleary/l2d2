pub enum Opcode {
    ANDPD,
    CMPNLESD,
    DIVQ,
    FADDP,
    FADDS,
    FDIVP,
    FILDLL,
    FLDL,
    FLD,
    FLDT,
    FLDZ,
    FMULL,
    FSTP,
    FSTPT,
    FUCOMIP,
    NOP,
    SETGE,
    SUBQ,

    PSRLDQ,
    PSUBB,
    VPANDN,
    VPCMPGTB,
    VPCMPISTRI,
    VPSUBB,

    LDDQU,
    MOVLPD,
    MOVHPD,
    PTEST,
    MOVBE,
    VPTEST,
    SETP,
    SUBSS,
    SUBW,
    NOTB,
    PMAXUB,
    VPBROADCASTB,
    NOTW,

    PXOR,
    PAND,
    PANDN,
    POR,
    PSUBQ,
    PSUBD,

    ADD,
    ADDB,
    ADDQ,

    AND,
    ANDB,
    CALLQ,

    CMP,
    CMPB,
    CMPL,
    CMPQ,

    JMP,
    JMPQ,
    SUB,

    MOVSLQ,
    MOVB,
    MOVD,
    MOVDQA,
    MOVABS,
    MOVAPS,
    MOVDQU,
    MOV,
    MOVL,
    MOVQ,
    MOVUPS,
    MOVZBL,
    MOVZWL,

    SHUFPS,
    PSHUFD,
    PADDD,
    PADDQ,

    PUSH,
    PUSHQ,
    POP,
    SHR,
    TEST,

    XOR,
    XORPS,
    ORPS,

    LEA,
    RETQ,
    IMUL,
    MUL,
    MULQ,
    DIV,
    DIVSD,

    ENDBR64,
    ORW,
    ORQ,

    ADC,

    INC,
    DEC,
    INCL,
    DECL,

    OR,
    SHL,
    SAR,

    JA,
    JAE,
    JB,
    JBE,
    JLE,
    JGE,
    JE,
    JNE,
    JS,
    JNS,
    JL,
    JO,
    JG,

    SETE,
    SETAE,
    SETNE,

    NOPL,
    NOPW,

    CMOVAE,
    CMOVLE,
    CMOVGE,
    CMOVE,
    CMOVNE,

    NEG,
    NEGL,

    NOT,

    XCHG,
    IN,
    OUT,

    VPADDQ,

    SUBL,
    ORL,
    COMISD,
    CVTTSS2SI,
    MOVSS,
    MULSS,
    CVTSS2SD,
    CVTSI2SDL,
    SETLE,
    MOVSQ,
    SHRX,
    SHLX,
    MULB,
    PREFETCHT1,
    SHLQ,
    TESTW,
    BTS,
    SETO,
    JP,
    CLD,

    ADDL,
    ADDR32,
    ADDSS,
    ADDSD,
    ADDW,
    AESENCLAST,
    AESKEYGENASSIST,
    ANDL,
    ANDQ,
    ANDW,
    BSF,
    BSR,
    BSWAP,
    BT,
    BTR,
    CLFLUSH,
    CLI,
    CLTD,
    CLTQ,
    CMC,
    CMOVA,
    CMOVBE,
    CMOVB,
    CMOVG,
    CMOVL,
    CMOVNS,
    CMOVS,
    CMPW,
    CPUID,
    CQTO,
    CVTSI2SD,
    CVTSI2SDQ,
    CVTSI2SS,
    CVTTSD2SI,
    CWTL,
    DATA16,
    DIVSS,
    DS,
    ENTERQ,
    FCOMP,
    FFREEP,
    FILDS,
    FLDS,
    FS,
    FWAIT,
    GS,
    ICEBP,
    IDIV,
    INSB,
    INT3,
    INT,
    JNO,
    LAHF,
    LCALL,
    LEAVEQ,
    LOCK,
    LOOPE,
    LOOPNE,
    MOVAPD,
    MOVHLPS,
    MOVLHPS,
    MOVSBL,
    MOVSB,
    MOVSBQ,
    MOVSD,
    MOVSWL,
    MOVSWQ,
    MOVUPD,
    MOVW,
    MOVZBQ,
    MULSD,
    ORB,
    OUTSB,
    OUTSL,
    PACKSSWB,
    PACKUSWB,
    PALIGNR,
    PAUSE,
    PCMPEQB,
    PCMPEQD,
    PCMPEQW,
    PCMPGTB,
    PINSRD,
    PINSRW,
    PMOVMSKB,
    PMULUDQ,
    POPFQ,
    PREFETCHNTA,
    PREFETCHT0,
    PSHUFB,
    PSHUFHW,
    PSHUFLW,
    PSLLD,
    PSLLDQ,
    PSLLQ,
    PSRAD,
    PSRLD,
    PSRLQ,
    PSRLW,
    PUNPCKHBW,
    PUNPCKHDQ,
    PUNPCKHQDQ,
    PUNPCKHWD,
    PUNPCKLBW,
    PUNPCKLDQ,
    PUNPCKLQDQ,
    PUNPCKLWD,
    PUSHFQ,
    RCLB,
    RCL,
    RDRAND,
    RDTSC,
    REPZ,
    ROLB,
    ROLL,
    ROL,
    ROR,
    SBB,
    SBBL,
    SCAS,
    SETA,
    SETBE,
    SETB,
    SETG,
    SETL,
    SETNO,
    SETNS,
    SETS,
    SHLD,
    SHRD,
    SHRQ,
    SHUFPD,
    SLDT,
    SS,
    STC,
    STD,
    STOS,
    STR,
    SUBSD,
    TESTB,
    TESTL,
    UCOMISD,
    UD2,
    UNPCKHPD,
    UNPCKLPD,
    VAESENC,
    VBROADCASTI128,
    VEXTRACTI128,
    VMOVAPS,
    VMOVD,
    VMOVDQA,
    VMOVDQU,
    VMOVQ,
    VMOVUPS,
    VPADDB,
    VPADDD,
    VPALIGNR,
    VPAND,
    VPBLENDD,
    VPBROADCASTQ,
    VPCLMULHQHQDQ,
    VPCLMULHQLQDQ,
    VPCLMULLQHQDQ,
    VPCLMULLQLQDQ,
    VPCMPEQB,
    VPCMPEQD,
    VPERM2I128,
    VPERMD,
    VPERMQ,
    VPEXTRQ,
    VPINSRD,
    VPMOVMSKB,
    VPMULUDQ,
    VPOR,
    VPROTD,
    VPSHUFB,
    VPSHUFD,
    VPSHUFHW,
    VPSHUFLW,
    VPSLLD,
    VPSLLDQ,
    VPSLLQ,
    VPSRLD,
    VPSRLDQ,
    VPSRLQ,
    VPSRLW,
    VPUNPCKHBW,
    VPUNPCKHDQ,
    VPUNPCKHQDQ,
    VPUNPCKLBW,
    VPUNPCKLDQ,
    VPUNPCKLQDQ,
    VPXOR,
    VZEROALL,
    VZEROUPPER,
    XLAT,

    ADCX,
    ANDNPD,
    BTCQ,
    BTSQ,
    CMPXCHG,
    COMISS,
    CVTSD2SS,
    DECQ,
    DIVL,
    FNSTCW,
    FNSTSW,
    FXAM,
    IDIVQ,
    INCQ,
    JNP,
    MOVHPS,
    MOVNTDQ,
    MOVSL,
    ORPD,
    PCMPESTRI,
    PCMPISTRI,
    PEXTRW,
    PMINUB,
    RCR,
    RORX,
    SETNP,
    SFENCE,
    SHRL,
    SYSCALL,
    TESTQ,
    TZCNT,
    UCOMISS,
    VMOVDQA64,
    VMOVDQU64,
    VMOVNTDQ,
    VPMINUB,
    XGETBV,

    XORPD,
}

pub fn get_opcode_count() -> u32 {
    Opcode::XORPD as u32 + 1
}

pub fn parse_opcode(s: &str) -> Option<Opcode> {
    let pieces = s.split_whitespace().collect::<Vec<_>>();
    if pieces.len() < 1 {
        return None;
    }
    Some(match pieces[0] {
        "sete" => Opcode::SETE,
        "setae" => Opcode::SETAE,
        "setne" => Opcode::SETNE,

        "xchg" => Opcode::XCHG,
        "in" => Opcode::IN,
        "out" => Opcode::OUT,

        "shufps" => Opcode::SHUFPS,
        "pshufd" => Opcode::PSHUFD,
        "psubq" => Opcode::PSUBQ,
        "psubd" => Opcode::PSUBD,
        "paddd" => Opcode::PADDD,
        "paddq" => Opcode::PADDQ,
        "por" => Opcode::POR,
        "pxor" => Opcode::PXOR,
        "pand" => Opcode::PAND,
        "pandn" => Opcode::PANDN,

        "not" => Opcode::NOT,
        "neg" => Opcode::NEG,
        "negl" => Opcode::NEGL,

        "nopl" => Opcode::NOPL,
        "nopw" => Opcode::NOPW,
        "addq" => Opcode::ADDQ,
        "vpaddq" => Opcode::VPADDQ,
        "xorps" => Opcode::XORPS,
        "orps" => Opcode::ORPS,
        "ja" => Opcode::JA,
        "jl" => Opcode::JL,
        "js" => Opcode::JS,
        "jns" => Opcode::JNS,
        "inc" => Opcode::INC,
        "incl" => Opcode::INCL,
        "dec" => Opcode::DEC,
        "decl" => Opcode::DECL,
        "shl" => Opcode::SHL,
        "sar" => Opcode::SAR,
        "retq" => Opcode::RETQ,
        "add" => Opcode::ADD,
        "addb" => Opcode::ADDB,
        "and" => Opcode::AND,
        "andb" => Opcode::ANDB,
        "callq" => Opcode::CALLQ,
        "cmp" => Opcode::CMP,
        "cmpb" => Opcode::CMPB,
        "cmpl" => Opcode::CMPL,
        "cmpq" => Opcode::CMPQ,
        "test" => Opcode::TEST,
        "shr" => Opcode::SHR,
        "jmp" => Opcode::JMP,
        "jmpq" => Opcode::JMPQ,
        "movb" => Opcode::MOVB,
        "movd" => Opcode::MOVD,
        "movdqa" => Opcode::MOVDQA,
        "movslq" => Opcode::MOVSLQ,
        "movdqu" => Opcode::MOVDQU,
        "movabs" => Opcode::MOVABS,
        "movaps" => Opcode::MOVAPS,
        "mov" => Opcode::MOV,
        "movl" => Opcode::MOVL,
        "movq" => Opcode::MOVQ,
        "movups" => Opcode::MOVUPS,
        "movzbl" => Opcode::MOVZBL,
        "movzwl" => Opcode::MOVZWL,
        "push" => Opcode::PUSH,
        "pushq" => Opcode::PUSHQ,
        "pop" => Opcode::POP,
        "or" => Opcode::OR,

        "xor" => Opcode::XOR,
        "lea" => Opcode::LEA,
        "sub" => Opcode::SUB,
        "adc" => Opcode::ADC,
        "imul" => Opcode::IMUL,
        "mul" => Opcode::MUL,
        "mulq" => Opcode::MULQ,
        "div" => Opcode::DIV,
        "divsd" => Opcode::DIVSD,

        "je" => Opcode::JE,
        "jle" => Opcode::JLE,
        "jg" => Opcode::JG,
        "jne" => Opcode::JNE,
        "jae" => Opcode::JAE,
        "jb" => Opcode::JB,
        "jbe" => Opcode::JBE,
        "jo" => Opcode::JO,
        "jge" => Opcode::JGE,

        "cmovae" => Opcode::CMOVAE,
        "cmove" => Opcode::CMOVE,
        "cmovne" => Opcode::CMOVNE,
        "cmovle" => Opcode::CMOVLE,
        "cmovge" => Opcode::CMOVGE,

        "addl" => Opcode::ADDL,
        "addr32" => Opcode::ADDR32,
        "addss" => Opcode::ADDSS,
        "addsd" => Opcode::ADDSD,
        "addw" => Opcode::ADDW,
        "aesenclast" => Opcode::AESENCLAST,
        "aeskeygenassist" => Opcode::AESKEYGENASSIST,
        "andl" => Opcode::ANDL,
        "andq" => Opcode::ANDQ,
        "andw" => Opcode::ANDW,
        "bsf" => Opcode::BSF,
        "bsr" => Opcode::BSR,
        "bswap" => Opcode::BSWAP,
        "bt" => Opcode::BT,
        "btr" => Opcode::BTR,
        "clflush" => Opcode::CLFLUSH,
        "cli" => Opcode::CLI,
        "cltd" => Opcode::CLTD,
        "cltq" => Opcode::CLTQ,
        "cmc" => Opcode::CMC,
        "cmova" => Opcode::CMOVA,
        "cmovbe" => Opcode::CMOVBE,
        "cmovb" => Opcode::CMOVB,
        "cmovg" => Opcode::CMOVG,
        "cmovl" => Opcode::CMOVL,
        "cmovns" => Opcode::CMOVNS,
        "cmovs" => Opcode::CMOVS,
        "cmpw" => Opcode::CMPW,
        "cpuid" => Opcode::CPUID,
        "cqto" => Opcode::CQTO,
        "cvtsi2sd" => Opcode::CVTSI2SD,
        "cvtsi2sdq" => Opcode::CVTSI2SDQ,
        "cvtsi2ss" => Opcode::CVTSI2SS,
        "cvttsd2si" => Opcode::CVTTSD2SI,
        "cwtl" => Opcode::CWTL,
        "data16" => Opcode::DATA16,
        "divss" => Opcode::DIVSS,
        "ds" => Opcode::DS,
        "enterq" => Opcode::ENTERQ,
        "fcomp" => Opcode::FCOMP,
        "ffreep" => Opcode::FFREEP,
        "filds" => Opcode::FILDS,
        "flds" => Opcode::FLDS,
        "fs" => Opcode::FS,
        "fwait" => Opcode::FWAIT,
        "gs" => Opcode::GS,
        "icebp" => Opcode::ICEBP,
        "idiv" => Opcode::IDIV,
        "insb" => Opcode::INSB,
        "int3" => Opcode::INT3,
        "int" => Opcode::INT,
        "jno" => Opcode::JNO,
        "lahf" => Opcode::LAHF,
        "lcall" => Opcode::LCALL,
        "leaveq" => Opcode::LEAVEQ,
        "lock" => Opcode::LOCK,
        "loope" => Opcode::LOOPE,
        "loopne" => Opcode::LOOPNE,
        "movapd" => Opcode::MOVAPD,
        "movhlps" => Opcode::MOVHLPS,
        "movlhps" => Opcode::MOVLHPS,
        "movsbl" => Opcode::MOVSBL,
        "movsb" => Opcode::MOVSB,
        "movsbq" => Opcode::MOVSBQ,
        "movsd" => Opcode::MOVSD,
        "movswl" => Opcode::MOVSWL,
        "movswq" => Opcode::MOVSWQ,
        "movupd" => Opcode::MOVUPD,
        "movw" => Opcode::MOVW,
        "movzbq" => Opcode::MOVZBQ,
        "mulsd" => Opcode::MULSD,
        "orb" => Opcode::ORB,
        "outsb" => Opcode::OUTSB,
        "outsl" => Opcode::OUTSL,
        "packsswb" => Opcode::PACKSSWB,
        "packuswb" => Opcode::PACKUSWB,
        "palignr" => Opcode::PALIGNR,
        "pause" => Opcode::PAUSE,
        "pcmpeqb" => Opcode::PCMPEQB,
        "pcmpeqd" => Opcode::PCMPEQD,
        "pcmpeqw" => Opcode::PCMPEQW,
        "pcmpgtb" => Opcode::PCMPGTB,
        "pinsrd" => Opcode::PINSRD,
        "pinsrw" => Opcode::PINSRW,
        "pmovmskb" => Opcode::PMOVMSKB,
        "pmuludq" => Opcode::PMULUDQ,
        "popfq" => Opcode::POPFQ,
        "prefetchnta" => Opcode::PREFETCHNTA,
        "prefetcht0" => Opcode::PREFETCHT0,
        "pshufb" => Opcode::PSHUFB,
        "pshufhw" => Opcode::PSHUFHW,
        "pshuflw" => Opcode::PSHUFLW,
        "pslld" => Opcode::PSLLD,
        "pslldq" => Opcode::PSLLDQ,
        "psllq" => Opcode::PSLLQ,
        "psrad" => Opcode::PSRAD,
        "psrld" => Opcode::PSRLD,
        "psrlq" => Opcode::PSRLQ,
        "psrlw" => Opcode::PSRLW,
        "punpckhbw" => Opcode::PUNPCKHBW,
        "punpckhdq" => Opcode::PUNPCKHDQ,
        "punpckhqdq" => Opcode::PUNPCKHQDQ,
        "punpckhwd" => Opcode::PUNPCKHWD,
        "punpcklbw" => Opcode::PUNPCKLBW,
        "punpckldq" => Opcode::PUNPCKLDQ,
        "punpcklqdq" => Opcode::PUNPCKLQDQ,
        "punpcklwd" => Opcode::PUNPCKLWD,
        "pushfq" => Opcode::PUSHFQ,
        "rclb" => Opcode::RCLB,
        "rcl" => Opcode::RCL,
        "rdrand" => Opcode::RDRAND,
        "rdtsc" => Opcode::RDTSC,
        "repz" => Opcode::REPZ,
        "rolb" => Opcode::ROLB,
        "roll" => Opcode::ROLL,
        "rol" => Opcode::ROL,
        "ror" => Opcode::ROR,
        "sbb" => Opcode::SBB,
        "sbbl" => Opcode::SBBL,
        "scas" => Opcode::SCAS,
        "seta" => Opcode::SETA,
        "setbe" => Opcode::SETBE,
        "setb" => Opcode::SETB,
        "setg" => Opcode::SETG,
        "setl" => Opcode::SETL,
        "setno" => Opcode::SETNO,
        "setns" => Opcode::SETNS,
        "sets" => Opcode::SETS,
        "shld" => Opcode::SHLD,
        "shrd" => Opcode::SHRD,
        "shrq" => Opcode::SHRQ,
        "shufpd" => Opcode::SHUFPD,
        "sldt" => Opcode::SLDT,
        "ss" => Opcode::SS,
        "stc" => Opcode::STC,
        "std" => Opcode::STD,
        "stos" => Opcode::STOS,
        "str" => Opcode::STR,
        "subsd" => Opcode::SUBSD,
        "testb" => Opcode::TESTB,
        "testl" => Opcode::TESTL,
        "ucomisd" => Opcode::UCOMISD,
        "ud2" => Opcode::UD2,
        "unpckhpd" => Opcode::UNPCKHPD,
        "unpcklpd" => Opcode::UNPCKLPD,
        "vaesenc" => Opcode::VAESENC,
        "vbroadcasti128" => Opcode::VBROADCASTI128,
        "vextracti128" => Opcode::VEXTRACTI128,
        "vmovaps" => Opcode::VMOVAPS,
        "vmovd" => Opcode::VMOVD,
        "vmovdqa" => Opcode::VMOVDQA,
        "vmovdqu" => Opcode::VMOVDQU,
        "vmovq" => Opcode::VMOVQ,
        "vmovups" => Opcode::VMOVUPS,
        "vpaddb" => Opcode::VPADDB,
        "vpaddd" => Opcode::VPADDD,
        "vpalignr" => Opcode::VPALIGNR,
        "vpand" => Opcode::VPAND,
        "vpblendd" => Opcode::VPBLENDD,
        "vpbroadcastq" => Opcode::VPBROADCASTQ,
        "vpclmulhqhqdq" => Opcode::VPCLMULHQHQDQ,
        "vpclmulhqlqdq" => Opcode::VPCLMULHQLQDQ,
        "vpclmullqhqdq" => Opcode::VPCLMULLQHQDQ,
        "vpclmullqlqdq" => Opcode::VPCLMULLQLQDQ,
        "vpcmpeqb" => Opcode::VPCMPEQB,
        "vpcmpeqd" => Opcode::VPCMPEQD,
        "vperm2i128" => Opcode::VPERM2I128,
        "vpermd" => Opcode::VPERMD,
        "vpermq" => Opcode::VPERMQ,
        "vpextrq" => Opcode::VPEXTRQ,
        "vpinsrd" => Opcode::VPINSRD,
        "vpmovmskb" => Opcode::VPMOVMSKB,
        "vpmuludq" => Opcode::VPMULUDQ,
        "vpor" => Opcode::VPOR,
        "vprotd" => Opcode::VPROTD,
        "vpshufb" => Opcode::VPSHUFB,
        "vpshufd" => Opcode::VPSHUFD,
        "vpshufhw" => Opcode::VPSHUFHW,
        "vpshuflw" => Opcode::VPSHUFLW,
        "vpslld" => Opcode::VPSLLD,
        "vpslldq" => Opcode::VPSLLDQ,
        "vpsllq" => Opcode::VPSLLQ,
        "vpsrld" => Opcode::VPSRLD,
        "vpsrldq" => Opcode::VPSRLDQ,
        "vpsrlq" => Opcode::VPSRLQ,
        "vpsrlw" => Opcode::VPSRLW,
        "vpunpckhbw" => Opcode::VPUNPCKHBW,
        "vpunpckhdq" => Opcode::VPUNPCKHDQ,
        "vpunpckhqdq" => Opcode::VPUNPCKHQDQ,
        "vpunpcklbw" => Opcode::VPUNPCKLBW,
        "vpunpckldq" => Opcode::VPUNPCKLDQ,
        "vpunpcklqdq" => Opcode::VPUNPCKLQDQ,
        "vpxor" => Opcode::VPXOR,
        "vzeroall" => Opcode::VZEROALL,
        "vzeroupper" => Opcode::VZEROUPPER,
        "xlat" => Opcode::XLAT,
        "xorpd" => Opcode::XORPD,
        "endbr64" => Opcode::ENDBR64,
        "orw" => Opcode::ORW,
        "orq" => Opcode::ORQ,

        "subl" => Opcode::SUBL,
        "orl" => Opcode::ORL,
        "comisd" => Opcode::COMISD,
        "cvttss2si" => Opcode::CVTTSS2SI,
        "movss" => Opcode::MOVSS,
        "mulss" => Opcode::MULSS,
        "cvtss2sd" => Opcode::CVTSS2SD,
        "setle" => Opcode::SETLE,
        "movsq" => Opcode::MOVSQ,
        "shrx" => Opcode::SHRX,
        "shlx" => Opcode::SHLX,
        "mulb" => Opcode::MULB,
        "prefetcht1" => Opcode::PREFETCHT1,
        "shlq" => Opcode::SHLQ,
        "testw" => Opcode::TESTW,
        "bts" => Opcode::BTS,
        "seto" => Opcode::SETO,
        "jp" => Opcode::JP,
        "cld" => Opcode::CLD,

        "andpd" => Opcode::ANDPD,
        "cmpnlesd" => Opcode::CMPNLESD,
        "divq" => Opcode::DIVQ,
        "faddp" => Opcode::FADDP,
        "fadds" => Opcode::FADDS,
        "fdivp" => Opcode::FDIVP,
        "fildll" => Opcode::FILDLL,
        "fldl" => Opcode::FLDL,
        "fld" => Opcode::FLD,
        "fldt" => Opcode::FLDT,
        "fldz" => Opcode::FLDZ,
        "fmull" => Opcode::FMULL,
        "fstp" => Opcode::FSTP,
        "fucomip" => Opcode::FUCOMIP,
        "nop" => Opcode::NOP,
        "setge" => Opcode::SETGE,
        "subq" => Opcode::SUBQ,

        "fstpt" => Opcode::FSTPT,

        "cvtsi2sdl" => Opcode::CVTSI2SDL,

        "adcx" => Opcode::ADCX,
        "andnpd" => Opcode::ANDNPD,
        "btcq" => Opcode::BTCQ,
        "btsq" => Opcode::BTSQ,
        "cmpxchg" => Opcode::CMPXCHG,
        "comiss" => Opcode::COMISS,
        "cvtsd2ss" => Opcode::CVTSD2SS,
        "decq" => Opcode::DECQ,
        "divl" => Opcode::DIVL,
        "fnstcw" => Opcode::FNSTCW,
        "fnstsw" => Opcode::FNSTSW,
        "fxam" => Opcode::FXAM,
        "idivq" => Opcode::IDIVQ,
        "incq" => Opcode::INCQ,
        "jnp" => Opcode::JNP,
        "movhps" => Opcode::MOVHPS,
        "movntdq" => Opcode::MOVNTDQ,
        "movsl" => Opcode::MOVSL,
        "orpd" => Opcode::ORPD,
        "pcmpestri" => Opcode::PCMPESTRI,
        "pcmpistri" => Opcode::PCMPISTRI,
        "pextrw" => Opcode::PEXTRW,
        "pminub" => Opcode::PMINUB,
        "rcr" => Opcode::RCR,
        "rorx" => Opcode::RORX,
        "setnp" => Opcode::SETNP,
        "sfence" => Opcode::SFENCE,
        "shrl" => Opcode::SHRL,
        "syscall" => Opcode::SYSCALL,
        "testq" => Opcode::TESTQ,
        "tzcnt" => Opcode::TZCNT,
        "ucomiss" => Opcode::UCOMISS,
        "vmovdqa64" => Opcode::VMOVDQA64,
        "vmovdqu64" => Opcode::VMOVDQU64,
        "vmovntdq" => Opcode::VMOVNTDQ,
        "vpminub" => Opcode::VPMINUB,
        "xgetbv" => Opcode::XGETBV,

        "psrldq" => Opcode::PSRLDQ,
        "psubb" => Opcode::PSUBB,
        "vpandn" => Opcode::VPANDN,
        "vpcmpgtb" => Opcode::VPCMPGTB,
        "vpcmpistri" => Opcode::VPCMPISTRI,
        "vpsubb" => Opcode::VPSUBB,

        "lddqu" => Opcode::LDDQU,
        "movlpd" => Opcode::MOVLPD,
        "movhpd" => Opcode::MOVHPD,
        "ptest" => Opcode::PTEST,
        "movbe" => Opcode::MOVBE,

        "vptest" => Opcode::VPTEST,
        "setp" => Opcode::SETP,
        "subss" => Opcode::SUBSS,
        "subw" => Opcode::SUBW,
        "notb" => Opcode::NOTB,
        "pmaxub" => Opcode::PMAXUB,
        "vpbroadcastb" => Opcode::VPBROADCASTB,
        "notw" => Opcode::NOTW,

        "notrack" => return parse_opcode(pieces[1]),
        "rep" => return parse_opcode(pieces[1]),
        "repnz" => return parse_opcode(pieces[1]),
        _ => {
            //eprintln!("Unknown opcode: {:?} in {:?}", pieces[0], s);
            eprintln!(
                "{:?} => Opcode::{},  // {:?}",
                pieces[0],
                pieces[0].to_uppercase(),
                s
            );
            return None;
        }
    })
}
