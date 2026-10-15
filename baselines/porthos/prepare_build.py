#!/usr/bin/env python3
"""Materialize and patch a disposable EzPC/Porthos tree at the pinned commit.

The tree is extracted from a git cache with ``git archive`` and every change is
applied to that disposable copy only.  Two build gates run here, before any
measurement: the round counter the patched binary uses, and the public-weight
kernels it links.  Their outcome is stamped into the build manifest, which the
runner reads before it publishes a round count.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LOCK = json.loads((ROOT / "ezpc.lock.json").read_text())
EIGEN_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz"
EIGEN_SHA256 = "d56fbad95abf993f8af608484729e3d87ef611dd85b3380a8bad1d5cbc373a57"


def file_digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _collapsed(text: str) -> str:
    """Runs of whitespace as single spaces."""
    return " ".join(text.split())


def _tolerant_span(source: str, old: str) -> tuple[int, int] | None:
    """The unique line span matching ``old`` modulo whitespace, or ``None``.

    The anchors below are upstream source fragments, and several of them carry
    tabs or trailing blanks that an editor normalizes away on the way into this
    file -- `connect.cpp`'s progress banner ends in a tab, for one.  A byte-exact
    anchor that silently stops matching is worse than no anchor: it makes a
    fresh checkout unpatchable while the script still looks correct.  Matching on
    collapsed whitespace keeps the "exactly one occurrence" guarantee, which is
    the property that actually matters.
    """
    needle = _collapsed(old)
    if not needle:
        return None
    lines = source.splitlines(keepends=True)
    spans = []
    for start in range(len(lines)):
        if not needle.startswith(_collapsed(lines[start])) or not lines[start].strip():
            continue
        for end in range(start + 1, len(lines) + 1):
            collapsed = _collapsed("".join(lines[start:end]))
            if collapsed == needle:
                spans.append((start, end))
                break
            if not needle.startswith(collapsed):
                break
    if len(spans) != 1:
        return None
    start, end = spans[0]
    offset = len("".join(lines[:start]))
    return offset, offset + len("".join(lines[start:end]))


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count == 1:
        path.write_text(source.replace(old, new))
        return
    if count == 0:
        span = _tolerant_span(source, old)
        if span is not None:
            start, end = span
            path.write_text(source[:start] + new + source[end:])
            return
    raise RuntimeError(f"{path}: expected one patch anchor, found {count}")


def _has_commit(cache: Path) -> bool:
    if not cache.exists():
        return False
    return subprocess.run(
        ["git", "-C", str(cache), "cat-file", "-e", f"{LOCK['commit']}^{{commit}}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def ensure_cache(cache: Path) -> None:
    """Make the pinned commit available in a bare cache, fetching only if absent."""
    if _has_commit(cache):
        return
    if not cache.exists():
        subprocess.run(["git", "init", "--bare", "--quiet", str(cache)], check=True)
    subprocess.run(["git", "-C", str(cache), "fetch", "--depth", "1",
                    LOCK["repository"], LOCK["commit"]], check=True)
    if not _has_commit(cache):
        raise RuntimeError(f"{cache}: pinned commit {LOCK['commit']} is missing")


def materialize(cache: Path, output: Path) -> None:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite disposable tree: {output}")
    output.mkdir(parents=True)
    archive_path = Path(tempfile.mkstemp(prefix="ghostmpc-porthos-", suffix=".tar")[1])
    try:
        with archive_path.open("wb") as archive:
            subprocess.run(["git", "-C", str(cache), "archive", LOCK["commit"]],
                           check=True, stdout=archive)
        with tarfile.open(archive_path) as archive:
            archive.extractall(output, filter="data")
    finally:
        archive_path.unlink(missing_ok=True)


def prepare_eigen(cache: Path, tree: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        temporary = cache.with_suffix(cache.suffix + ".partial")
        urlretrieve(EIGEN_URL, temporary)
        temporary.replace(cache)
    if file_digest(cache) != EIGEN_SHA256:
        raise RuntimeError(f"Eigen archive hash mismatch: {cache}")
    with tarfile.open(cache) as archive:
        archive.extractall(tree / "Porthos", filter="data")
    extracted = tree / "Porthos/eigen-3.3.7"
    target = tree / "Porthos/lib_eigen"
    if target.exists() or not extracted.is_dir():
        raise RuntimeError("unexpected Eigen extraction layout")
    extracted.rename(target)


def patch_athos(tree: Path) -> None:
    process = tree / "Athos/ONNXCompiler/process_onnx.py"
    replace_once(
        process,
        '''                if node.doc_string == "MPC_MODEL_WEIGHTS":
                    self.party = AST.Party.SERVER
                else:
                    self.party = AST.Party.CLIENT
            elif isinstance(node, TensorProto):  # initializers
                self.shape = list(node.dims)
                self.data_type = node.data_type
                self.party = AST.Party.SERVER
''',
        '''                if node.doc_string in ("MPC_MODEL_WEIGHTS", "MPC_PUBLIC_MODEL_WEIGHT"):
                    self.party = AST.Party.SERVER
                    self.is_secret = False
                else:
                    self.party = AST.Party.CLIENT
                    self.is_secret = True
            elif isinstance(node, TensorProto):  # initializers
                self.shape = list(node.dims)
                self.data_type = node.data_type
                self.party = AST.Party.SERVER
                self.is_secret = node.doc_string != "MPC_PUBLIC_MODEL_WEIGHT"
''')
    nodes = tree / "Athos/ONNXCompiler/ONNXNodesAST.py"
    replace_once(
        nodes,
        '''        return AST.Input(
            node.shape, onnx2seedot(node.data_type), inputByParty=node.party
        )
''',
        '''        return AST.Input(
            node.shape, onnx2seedot(node.data_type), isSecret=node.is_secret,
            inputByParty=node.party
        )
''')
    replace_once(
        nodes,
        '''        [zPadHLeft, zPadHRight, zPadWLeft, zPadWRight] = (
            node.attrs["pads"] if "pads" in node.attrs else [0, 0, 0, 0]
        )
''',
        '''        # ONNX pads: H_begin, W_begin, H_end, W_end.
        [zPadHLeft, zPadWLeft, zPadHRight, zPadWRight] = (
            node.attrs["pads"] if "pads" in node.attrs else [0, 0, 0, 0]
        )
''')
    ir = tree / "Athos/SeeDot/IR/IRBuilderCSF.py"
    replace_once(
        ir,
        '''        funcCallName += "Wrapper"

        funcCall = IR.FuncCall(funcCallName, funcCallArgsDict)
''',
        '''        funcCallName += "Wrapper"
        if not node.expr2.type.isSecret:
            funcCallName = "Ghost" + funcCallName.replace(
                "Wrapper", "PublicWrapper")

        funcCall = IR.FuncCall(funcCallName, funcCallArgsDict)
''')
    replace_once(
        ir,
        '''        funcCall = IR.FuncCall("MatMul2D", funcCallArgsDict)
''',
        '''        funcCall = IR.FuncCall(
            "GhostMatMulPublic" if (not typ_1.isSecret or not typ_2.isSecret)
            else "MatMul2D", funcCallArgsDict)
''')
    replace_once(
        ir,
        '''                funcName = (
                    "ElemWiseActModelVectorMult"
                    if isMulOptimised
                    else "ElemWiseSecretSharedVectorMult"
                )
''',
        '''                if not node.expr1.type.isSecret or not node.expr2.type.isSecret:
                    funcName = "GhostElemWiseMulPublic"
                else:
                    funcName = (
                        "ElemWiseActModelVectorMult"
                        if isMulOptimised
                        else "ElemWiseSecretSharedVectorMult"
                    )
''')
    replace_once(
        ir,
        '''            out_prog = IRUtil.generateBroadcastLoopBOp(
                expr_1, inp1_shape, expr_2, inp2_shape, out_arr, op_ir
            )
''',
        '''            if (node.expr1.type.isSecret != node.expr2.type.isSecret
                    and op in (AST.Operators.ADD, AST.Operators.SUB)):
                _, mask_1, mask_2 = Util.getBroadcastShapes(inp1_shape, inp2_shape)
                out_iters = self.getTempIterators(len(outputShape))
                elem_1 = IRUtil.addIndex(
                    expr_1, IRUtil.getMaskedIters(mask_1, out_iters, inp1_shape))
                elem_2 = IRUtil.addIndex(
                    expr_2, IRUtil.getMaskedIters(mask_2, out_iters, inp2_shape))
                out_elem = IRUtil.addIndex(out_arr, out_iters)
                if op == AST.Operators.ADD:
                    secret, clear = ((elem_1, elem_2) if node.expr1.type.isSecret
                                     else (elem_2, elem_1))
                    func_name = "GhostAddPublicScalar"
                elif node.expr1.type.isSecret:
                    secret, clear = elem_1, elem_2
                    func_name = "GhostSecretMinusPublicScalar"
                else:
                    clear, secret = elem_1, elem_2
                    func_name = "GhostPublicMinusSecretScalar"
                mixed = IR.FuncCall(
                    func_name, OrderedDict([(secret, "secret"), (clear, "clear")]))
                out_prog = IR.Prog(IRUtil.loop(
                    outputShape, out_iters, [IR.Assn(out_elem, mixed)]))
            else:
                out_prog = IRUtil.generateBroadcastLoopBOp(
                    expr_1, inp1_shape, expr_2, inp2_shape, out_arr, op_ir)
''')
    replace_once(
        ir,
        '''            funcCall = IR.FuncCall("ClearMemSecret1", argsDict)
            out_prog = IRUtil.prog_merge(out_prog, IRUtil.Prog([funcCall]))
            argsDict = OrderedDict()
            argsDict[IR.Int(flat_size, 32)] = "size"
            argsDict[inp2_arr_flat] = "A"
            funcCall = IR.FuncCall("ClearMemSecret1", argsDict)
''',
        '''            funcCall = IR.FuncCall(
                "ClearMemSecret1" if node.expr1.type.isSecret
                else "ClearMemPublic1", argsDict)
            out_prog = IRUtil.prog_merge(out_prog, IRUtil.Prog([funcCall]))
            argsDict = OrderedDict()
            argsDict[IR.Int(flat_size, 32)] = "size"
            argsDict[inp2_arr_flat] = "A"
            funcCall = IR.FuncCall(
                "ClearMemSecret1" if node.expr2.type.isSecret
                else "ClearMemPublic1", argsDict)
''')
    replace_once(
        ir,
        '''                curFuncName = "ScaleUp"
''',
        '''                curFuncName = ("ScaleUp" if typeOfExprToScale.isSecret
                                   else "GhostScaleUpPublic")
''')
    replace_once(
        ir,
        '''        prog_2 = IRUtil.prog_merge(IR.Prog([IR.Decl(expr_2.idf, typ_2)]), prog_2)
''',
        '''        prog_2 = IRUtil.prog_merge(IR.Prog([
            IR.Decl(expr_2.idf, typ_2, isSecret=typ_2.isSecret)]), prog_2)
''')
    codegen = tree / "Athos/SeeDot/Codegen/EzPC.py"
    replace_once(
        codegen,
        '''        inputByPartyStr = ir.inputByParty.name
        assert inputByPartyStr in ["SERVER", "CLIENT"]
''',
        '''        inputByPartyStr = ("ALL" if not ir.isSecret
                           else ir.inputByParty.name)
        assert inputByPartyStr in ["SERVER", "CLIENT", "ALL"]
''')


def patch_porthos(tree: Path) -> None:
    globals_h = tree / "Porthos/src/globals.h"
    replace_once(
        globals_h, "#include <limits.h>\n", "#include <limits.h>\n#include <cstdint>\n")
    replace_once(globals_h, "#define NO_CORES 4\n", "#define NO_CORES 1\n")
    # Fixed repository keys remain unused: each run receives a freshly generated
    # key directory as argv[3]. Keep the macro only for source compatibility.
    replace_once(
        globals_h,
        '#define FIXED_KEY_AES "43739841701238781571456410093f43"\n',
        '#define FIXED_KEY_AES "UNUSED_SESSION_KEYS_COME_FROM_ARGV"\n')
    replace_once(
        globals_h,
        'extern uint32_t FLOAT_PRECISION;\n',
        '''extern uint32_t FLOAT_PRECISION;
bool GhostPorthosNearestEnabled();
''')
    sockets_h = tree / "Porthos/src/basicSockets.h"
    replace_once(
        sockets_h,
        "#include <vector>\n",
        '#include <vector>\n#include <atomic>\n'
        '#include "causal_round_counter.h"\n')
    replace_once(
        sockets_h,
        '''	size_t bytesSent = 0;
	size_t bytesReceived = 0;
	size_t numberOfSends = 0;
	size_t numberOfRecvs = 0;
	bool measurement = false;
''',
        '''	std::atomic<size_t> bytesSent{0};
	std::atomic<size_t> bytesReceived{0};
	std::atomic<size_t> numberOfSends{0};
	std::atomic<size_t> numberOfRecvs{0};
	std::atomic<bool> measurement{false};
	// The round counter lives in the repository header so that the rule has one
	// statement and `test_round_counter.cpp` can drive the same code the
	// measured binary runs.
	ghostmpc_porthos::SerialRoundCounter roundCounter;
''')
    replace_once(
        sockets_h,
        '''	void setMeasurement(bool a)
	{
		measurement = a;
	}
''',
        '''	void setMeasurement(bool a)
	{
		measurement = a;
		roundCounter.arm(a);
	}
''')
    replace_once(
        sockets_h,
        '''	void reset()
	{
		bytesSent = 0;
		bytesReceived = 0;
		numberOfSends = 0;
		numberOfRecvs = 0;
		measurement = false;
''',
        '''	void reset()
	{
		bytesSent = 0;
		bytesReceived = 0;
		numberOfSends = 0;
		numberOfRecvs = 0;
		roundCounter.reset();
		measurement = false;
''')
    replace_once(
        sockets_h,
        '''	void incrementSent(size_t size)
	{
		if (measurement)
		{
			bytesSent += size;
			numberOfSends++;
		}
	}

	void incrementRecv(size_t size)
	{
		if (measurement)
		{
			bytesReceived += size;
			numberOfRecvs++;
		}
	}

	size_t getSent() {return bytesSent;}
	size_t getRecv() {return bytesReceived;}
	size_t getRoundsSent() {return numberOfSends;}
	size_t getRoundsRecv() {return numberOfRecvs;}
	bool getMeasurement() {return measurement;}
''',
        '''	void incrementSent(size_t size)
	{
		if (measurement.load(std::memory_order_relaxed))
		{
			bytesSent.fetch_add(size, std::memory_order_relaxed);
			numberOfSends.fetch_add(1, std::memory_order_relaxed);
			roundCounter.sent();
		}
	}

	void incrementRecv(size_t size)
	{
		if (measurement.load(std::memory_order_relaxed))
		{
			bytesReceived.fetch_add(size, std::memory_order_relaxed);
			numberOfRecvs.fetch_add(1, std::memory_order_relaxed);
			roundCounter.received();
		}
	}

	void beginExchange() {roundCounter.begin_exchange();}
	void endExchange() {roundCounter.end_exchange();}

	size_t getSent() {return bytesSent.load(std::memory_order_relaxed);}
	size_t getRecv() {return bytesReceived.load(std::memory_order_relaxed);}
	size_t getRounds() {return roundCounter.rounds();}
	size_t getRoundsSent() {return numberOfSends.load(std::memory_order_relaxed);}
	size_t getRoundsRecv() {return numberOfRecvs.load(std::memory_order_relaxed);}
	bool getMeasurement() {return measurement.load(std::memory_order_relaxed);}
''')
    tools_cpp = tree / "Porthos/src/tools.cpp"
    tools_source = tools_cpp.read_text()
    for old, new, expected in (
        ("floatToMyType(r[i] % 2)",
         "(GhostPorthosNearestEnabled() ? (r[i] % 2) : floatToMyType(r[i] % 2))",
         6),
        ("floatToMyType(vec[i])",
         "(GhostPorthosNearestEnabled() ? vec[i] : floatToMyType(vec[i]))",
         6),
    ):
        count = tools_source.count(old)
        if count != expected:
            raise RuntimeError(
                f"{tools_cpp}: expected {expected} raw-bit anchors, found {count}")
        tools_source = tools_source.replace(old, new)
    tools_cpp.write_text(tools_source)

    functionalities = tree / "Porthos/src/Functionalities.cpp"
    functionalities_source = functionalities.read_text()
    old = "j = floatToMyType(1);"
    if functionalities_source.count(old) != 2:
        raise RuntimeError(f"{functionalities}: raw selector anchors changed")
    functionalities_source = functionalities_source.replace(
        old, "j = GhostPorthosNearestEnabled() ? 1 : floatToMyType(1);")
    old = "funcDotProductMPC(theta_shares, LSB_shares, prod, size);"
    new = '''if (GhostPorthosNearestEnabled())
        funcDotProductMPC(theta_shares, LSB_shares, prod, size, 0, false);
    else
        funcDotProductMPC(theta_shares, LSB_shares, prod, size);'''
    if functionalities_source.count(old) != 1:
        raise RuntimeError(f"{functionalities}: MSB product anchor changed")
    functionalities_source = functionalities_source.replace(old, new)
    old = '''assert(THREE_PC && "funcSelectShares3PC called in non-3PC mdoe");
\tfuncDotProductMPC(a, b, c, size);'''
    new = '''assert(THREE_PC && "funcSelectShares3PC called in non-3PC mdoe");
\tif (GhostPorthosNearestEnabled())
        funcDotProductMPC(a, b, c, size, 0, false);
    else
        funcDotProductMPC(a, b, c, size);'''
    if functionalities_source.count(old) != 1:
        raise RuntimeError(f"{functionalities}: selector product anchor changed")
    functionalities.write_text(functionalities_source.replace(old, new))
    # The three PARALLEL_COMM sites below run a send and a receive on
    # two threads, so no local order exists between them and the socket-level
    # rule ("a round opens on the first receive after a non-receive") would
    # count each 0 or 1 times depending on the scheduler.  Each is one
    # simultaneous exchange -- one round -- and announces itself before the
    # spawn, which happens-before every socket call the threads then make.  The
    # five other PARALLEL_COMM blocks are parallel *sends* with no receive: a
    # send is not a round, and concurrent sends agree on the flag they write.
    for sender, receiver in (
        ("threads[0] = thread(sendTwoVectors<porthosSecretType>, ref(E), "
         "ref(F), adversary(partyNum), size_left, size_right);",
         "threads[1] = thread(receiveTwoVectors<porthosSecretType>, "
         "ref(temp_E), ref(temp_F), adversary(partyNum), size_left, "
         "size_right);"),
        ("threads[0] = thread(sendTwoVectors<porthosSecretType>, ref(E), "
         "ref(F), adversary(partyNum), size, size);",
         "threads[1] = thread(receiveTwoVectors<porthosSecretType>, "
         "ref(temp_E), ref(temp_F), adversary(partyNum), size, size);"),
        ("threads[0] = thread(sendVector<porthosSecretType>, ref(c), "
         "adversary(partyNum), size);",
         "threads[1] = thread(receiveVector<porthosSecretType>, ref(temp), "
         "adversary(partyNum), size);"),
    ):
        body = (f"\t\tthread *threads = new thread[2];\n\n"
                f"\t\t{sender}\n\t\t{receiver}\n\n"
                f"\t\tfor (int i = 0; i < 2; i++)\n"
                f"\t\t\tthreads[i].join();\n\n"
                f"\t\tdelete[] threads;\n")
        replace_once(
            functionalities, body,
            "\t\tcommObject.beginExchange();\n" + body +
            "\t\tcommObject.endExchange();\n")
    connect = tree / "Porthos/src/connect.cpp"
    replace_once(
        connect,
        'extern void error(string str);\n',
        '''extern void error(string str);

static int ghostmpc_connect_port_offset() {
  const char* raw = std::getenv("GHOSTMPC_CONNECT_PORT_OFFSET");
  return raw ? std::atoi(raw) : 0;
}
''')
    # The barrier is one round: every party sends to and receives from both
    # peers at once.  Its send and receive run on separate threads, so it is
    # announced like the PARALLEL_COMM exchanges rather than inferred.
    replace_once(
        connect,
        '''	char* toSend = new char[length+1];
	memset(toSend, '0', length+1);
	vector<thread *> threads;
''',
        '''	char* toSend = new char[length+1];
	memset(toSend, '0', length+1);
	commObject.beginExchange();
	vector<thread *> threads;
''')
    replace_once(
        connect,
        '''	threads.clear();
	delete[] toSend;
''',
        '''	threads.clear();
	commObject.endExchange();
	delete[] toSend;
''')
    replace_once(
        connect,
        '''		ports[2 * i] = 32000 + i*NUM_OF_PARTIES + partyNum + 100*instanceID;
		ports[2 * i + 1] = 32000 + partyNum*NUM_OF_PARTIES + i + 100*instanceID;
''',
        '''		ports[2 * i] = 32000 + i*NUM_OF_PARTIES + partyNum + 100*instanceID;
		ports[2 * i + 1] = 32000 + partyNum*NUM_OF_PARTIES + i +
			100*instanceID + ghostmpc_connect_port_offset();
''')
    replace_once(
        connect,
        '''	cout << "------------------------------------" << endl;
	commObject.reset();
}
''',
        '''	cout << "------------------------------------" << endl;
	cout << "GHOSTMPC_PORTHOS_COMM {\\\"party\\\":" << partyNum
		 << ",\\\"bytes_sent\\\":" << commObject.getSent()
		 << ",\\\"bytes_received\\\":" << commObject.getRecv()
		 << ",\\\"send_calls\\\":" << commObject.getRoundsSent()
		 << ",\\\"recv_calls\\\":" << commObject.getRoundsRecv()
		 << ",\\\"rounds\\\":" << commObject.getRounds()
		 << "}" << endl;
	commObject.reset();
}
''')
    # CLOCK_REALTIME can step backwards under NTP in the middle of a request,
    # and clock() resolves to CLOCKS_PER_SEC.  Both timers move to the
    # monotonic clocks; neither shifts a measured quantity systematically.
    tools = tree / "Porthos/src/tools.cpp"
    replace_once(
        tools,
        '''void start_time()
{
''',
        '''static timespec ghostmpcCpuStart;

void start_time()
{
''')
    replace_once(
        tools,
        '''	tStart = clock();
	clock_gettime(CLOCK_REALTIME, &requestStart);
	alreadyMeasuringTime = true;
''',
        '''	tStart = clock();
	clock_gettime(CLOCK_MONOTONIC, &requestStart);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ghostmpcCpuStart);
	alreadyMeasuringTime = true;
''')
    replace_once(
        tools,
        '''	clock_gettime(CLOCK_REALTIME, &requestEnd);
#ifdef RUNTIME_DETAILED
''',
        '''	clock_gettime(CLOCK_MONOTONIC, &requestEnd);
	timespec ghostmpcCpuEnd;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ghostmpcCpuEnd);
	const double ghostmpcWall = diff(requestStart, requestEnd);
	const double ghostmpcCpu = diff(ghostmpcCpuStart, ghostmpcCpuEnd);
	const std::streamsize ghostmpcPrecision = cout.precision(12);
	cout << "GHOSTMPC_PORTHOS_TIME {\\\"party\\\":" << partyNum
		 << ",\\\"wall_seconds\\\":" << ghostmpcWall
		 << ",\\\"cpu_seconds\\\":" << ghostmpcCpu << "}" << endl;
	cout.precision(ghostmpcPrecision);
#ifdef RUNTIME_DETAILED
''')
    shutil.copy2(ROOT / "ghostmpc_public_kernels.h", tree / "Porthos/src")
    shutil.copy2(ROOT / "causal_round_counter.h", tree / "Porthos/src")
    shutil.copy2(ROOT / "porthos_runtime_support.h", tree / "Porthos/src")
    shutil.copy2(ROOT / "ghostmpc_porthos_extension.cpp", tree / "Porthos/src")
    ezpc_functionalities = tree / "Porthos/src/EzPCFunctionalities.cpp"
    replace_once(
        ezpc_functionalities,
        '''\tfuncRELUMPC(inArr, outArr, size);
\tif (doTruncation && ((partyNum==PARTY_A) || (partyNum==PARTY_B))){
\t\tfuncTruncate2PC(outArr, sf, size);
\t}
''',
        '''\tfuncRELUMPC(inArr, outArr, size);
\tif (doTruncation && GhostPorthosNearestEnabled())
\t\tGhostTruncateNearest(outArr, sf, size);
\telse if (doTruncation && ((partyNum==PARTY_A) || (partyNum==PARTY_B)))
\t\tfuncTruncate2PC(outArr, sf, size);
''')
    replace_once(
        ezpc_functionalities,
        '''\tif ((partyNum==PARTY_A) || (partyNum==PARTY_B))
\t\tfuncTruncate2PC(arr, sf, s1);
''',
        '''\tif (GhostPorthosNearestEnabled())
\t\tGhostTruncateNearest(arr, sf, s1);
\telse if ((partyNum==PARTY_A) || (partyNum==PARTY_B))
\t\tfuncTruncate2PC(arr, sf, s1);
''')
    replace_once(
        ezpc_functionalities,
        '''\t\t\t\t\t//IMP NOTE : The local division should always be signed division.
\t\t\t\t\t//TODO : For now doing local truncation : but this will introduce error
\t\t\t\t\tif (partyNum == PARTY_A)
\t\t\t\t\t\tfilterAvg[rowIdx] = static_cast<porthosSecretType>((static_cast<porthosSignedSecretType>(curFilterSum))/(ksizeH*ksizeW));

\t\t\t\t\tif (partyNum == PARTY_B)
\t\t\t\t\t\tfilterAvg[rowIdx] = -static_cast<porthosSecretType>((static_cast<porthosSignedSecretType>(-curFilterSum))/(ksizeH*ksizeW));
''',
        '''\t\t\t\t\tif (GhostPorthosNearestEnabled()) {
\t\t\t\t\t\t// Delay public division until every window sum is available. The
\t\t\t\t\t\t// reciprocal multiplication is local on additive shares; its only
\t\t\t\t\t\t// interactive step is the secure nearest truncation below.
\t\t\t\t\t\tfilterAvg[rowIdx] = curFilterSum;
\t\t\t\t\t} else if (partyNum == PARTY_A) {
\t\t\t\t\t\tfilterAvg[rowIdx] = static_cast<porthosSecretType>((static_cast<porthosSignedSecretType>(curFilterSum))/(ksizeH*ksizeW));
\t\t\t\t\t} else if (partyNum == PARTY_B) {
\t\t\t\t\t\tfilterAvg[rowIdx] = -static_cast<porthosSecretType>((static_cast<porthosSignedSecretType>(-curFilterSum))/(ksizeH*ksizeW));
\t\t\t\t\t}
''')
    replace_once(
        ezpc_functionalities,
        '''\tfor(int n=0;n<N;n++){
\t\tfor(int c=0;c<C;c++){
\t\t\tfor(int h=0;h<H;h++){
\t\t\t\tfor(int w=0;w<W;w++){
\t\t\t\t\tint iidx = n*C*H*W + c*H*W + h*W + w;
\t\t\t\t\toutArr[n][h][w][c] = filterAvg[iidx];
''',
        '''\tif (GhostPorthosNearestEnabled()) {
\t\tconst porthosSecretType divisor = ksizeH * ksizeW;
\t\tconst porthosSecretType reciprocal =
\t\t\t((porthosSecretType(1) << 24) + divisor / 2) / divisor;
\t\tif (PRIMARY)
\t\t\tfor (int i = 0; i < rows; ++i) filterAvg[i] *= reciprocal;
\t\tGhostTruncateNearest(filterAvg, 24, rows);
\t}

\tfor(int n=0;n<N;n++){
\t\tfor(int c=0;c<C;c++){
\t\t\tfor(int h=0;h<H;h++){
\t\t\t\tfor(int w=0;w<W;w++){
\t\t\t\t\tint iidx = n*C*H*W + c*H*W + h*W + w;
\t\t\t\t\toutArr[n][h][w][c] = filterAvg[iidx];
''')
    library = tree / "Athos/TFEzPCLibrary/Library64_porthos.ezpc"
    public_clear32 = '''extern void ClearMemPublic(int32_pl x);
extern void ClearMemPublic1(int32_pl s, int32_pl[s] x);
extern void ClearMemPublic2(int32_pl s1, int32_pl s2, int32_pl[s1][s2] arr);
extern void ClearMemPublic3(int32_pl s1, int32_pl s2, int32_pl s3, int32_pl[s1][s2][s3] arr);
extern void ClearMemPublic4(int32_pl s1, int32_pl s2, int32_pl s3, int32_pl s4, int32_pl[s1][s2][s3][s4] arr);
extern void ClearMemPublic5(int32_pl s1, int32_pl s2, int32_pl s3, int32_pl s4, int32_pl s5, int32_pl[s1][s2][s3][s4][s5] arr);'''
    public_clear64 = public_clear32.replace(
        "ClearMemPublic(int32_pl x)", "ClearMemPublic(int64_pl x)").replace(
            "int32_pl[", "int64_pl[")
    replace_once(library, public_clear32, public_clear64)
    declarations = r'''

(* GhostMPC generic public-weight extension. Public MACs are local on P0/P1. *)
extern void GhostMatMulPublic(int32_pl i, int32_pl j, int32_pl k, int64_al[i][j] A, int64_pl[j][k] B, int64_al[i][k] C, bool_pl modelIsA);
extern void GhostElemWiseMulPublic(int32_pl s1, int64_al[s1] A, int64_pl[s1] B, int64_al[s1] C);
extern int64_al GhostAddPublicScalar(int64_al secret, int64_pl clear);
extern int64_al GhostSecretMinusPublicScalar(int64_al secret, int64_pl clear);
extern int64_al GhostPublicMinusSecretScalar(int64_al secret, int64_pl clear);
extern void GhostScaleUpPublic1(int32_pl s1, int64_pl[s1] A, int32_pl sf);
extern void GhostScaleUpPublic2(int32_pl s1, int32_pl s2, int64_pl[s1][s2] A, int32_pl sf);
extern void GhostScaleUpPublic3(int32_pl s1, int32_pl s2, int32_pl s3, int64_pl[s1][s2][s3] A, int32_pl sf);
extern void GhostScaleUpPublic4(int32_pl s1, int32_pl s2, int32_pl s3, int32_pl s4, int64_pl[s1][s2][s3][s4] A, int32_pl sf);
extern void GhostScaleUpPublic5(int32_pl s1, int32_pl s2, int32_pl s3, int32_pl s4, int32_pl s5, int64_pl[s1][s2][s3][s4][s5] A, int32_pl sf);
extern void HardSigmoid(int32_pl s1, int64_al[s1] inArr, int64_al[s1] outArr, int32_pl sf, bool_pl doTruncation);
extern void GhostConv2DPublicWrapper(int32_pl N, int32_pl H, int32_pl W, int32_pl CI, int32_pl FH, int32_pl FW, int32_pl CO, int32_pl pT, int32_pl pB, int32_pl pL, int32_pl pR, int32_pl sH, int32_pl sW, int64_al[N][H][W][CI] A, int64_pl[FH][FW][CI][CO] B, int64_al[N][H][W][CO] C);
extern void GhostConv2DGroupPublicWrapper(int32_pl N, int32_pl H, int32_pl W, int32_pl CI, int32_pl FH, int32_pl FW, int32_pl CO, int32_pl pT, int32_pl pB, int32_pl pL, int32_pl pR, int32_pl sH, int32_pl sW, int32_pl G, int64_al[N][H][W][CI] A, int64_pl[FH][FW][CI/G][CO] B, int64_al[N][H][W][CO] C);
'''
    library.write_text(library.read_text() + declarations)
    replace_once(
        ezpc_functionalities,
        '''	synchronize(2000000);
''',
        '''	GhostSyncBegin();
	synchronize(2000000);
	GhostSyncEnd();
''')
    header = tree / "Porthos/src/EzPCFunctionalities.h"
    header.write_text(header.read_text() + r'''

porthosSecretType GhostAddPublicScalar(porthosSecretType, int64_t);
porthosSecretType GhostSecretMinusPublicScalar(porthosSecretType, int64_t);
porthosSecretType GhostPublicMinusSecretScalar(porthosSecretType, int64_t);
bool GhostPorthosNearestEnabled();
void GhostTruncateNearest(std::vector<porthosSecretType>&, std::size_t,
  std::size_t);
// The offline and per-request-barrier phase markers.
void GhostOfflineBegin();
void GhostOfflineEnd();
void GhostSyncBegin();
void GhostSyncEnd();
// The 1:1 verification score, from Porthos' own secret-by-secret product,
// because Athos cannot lower one from ONNX for any of its targets.
void GhostScore(int32_t, std::vector<porthosSecretType>&,
  std::vector<porthosSecretType>&, int32_t, std::vector<porthosSecretType>&);
void GhostScaleUpPublic1(int32_t, std::vector<int64_t>&, int32_t);
void GhostScaleUpPublic2(int32_t, int32_t,
  std::vector<std::vector<int64_t>>&, int32_t);
void GhostScaleUpPublic3(int32_t, int32_t, int32_t,
  std::vector<std::vector<std::vector<int64_t>>>&, int32_t);
void GhostScaleUpPublic4(int32_t, int32_t, int32_t, int32_t,
  std::vector<std::vector<std::vector<std::vector<int64_t>>>>&, int32_t);
void GhostScaleUpPublic5(int32_t, int32_t, int32_t, int32_t, int32_t,
  std::vector<std::vector<std::vector<std::vector<std::vector<int64_t>>>>>&,
  int32_t);
void HardSigmoid(int32_t, std::vector<porthosSecretType>&,
  std::vector<porthosSecretType>&, int32_t, bool);
void ClearMemPublic1(int64_t, std::vector<int64_t>&);
void ClearMemPublic2(int64_t, int64_t, std::vector<std::vector<int64_t>>&);
void ClearMemPublic3(int64_t, int64_t, int64_t,
  std::vector<std::vector<std::vector<int64_t>>>&);
void ClearMemPublic4(int64_t, int64_t, int64_t, int64_t,
  std::vector<std::vector<std::vector<std::vector<int64_t>>>>&);
void ClearMemPublic5(int64_t, int64_t, int64_t, int64_t, int64_t,
  std::vector<std::vector<std::vector<std::vector<std::vector<int64_t>>>>>&);
void GhostMatMulPublic(int32_t, int32_t, int32_t,
  std::vector<std::vector<porthosSecretType>>&,
  std::vector<std::vector<int64_t>>&,
  std::vector<std::vector<porthosSecretType>>&, bool);
void GhostElemWiseMulPublic(int32_t, std::vector<porthosSecretType>&,
  std::vector<int64_t>&, std::vector<porthosSecretType>&);
void GhostConv2DPublicWrapper(int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,
  int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,
  std::vector<std::vector<std::vector<std::vector<porthosSecretType>>>>&,
  std::vector<std::vector<std::vector<std::vector<int64_t>>>>&,
  std::vector<std::vector<std::vector<std::vector<porthosSecretType>>>>&);
void GhostConv2DGroupPublicWrapper(int32_t,int32_t,int32_t,int32_t,int32_t,
  int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,int32_t,
  std::vector<std::vector<std::vector<std::vector<porthosSecretType>>>>&,
  std::vector<std::vector<std::vector<std::vector<int64_t>>>>&,
  std::vector<std::vector<std::vector<std::vector<porthosSecretType>>>>&);
''')


def compiler_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
    except OSError:
        return None
    return result.stdout.splitlines()[0] if result.stdout else None


ROUND_COUNTER_TEST = ROOT / "test_round_counter.cpp"
PUBLIC_KERNEL_TEST = ROOT / "test_public_kernels.cpp"


def _compile_and_run(source: Path, prefix: str) -> str:
    """Compile ``source`` against this directory's headers and run it."""
    compiler = os.environ.get("CXX", "c++")
    binary = Path(tempfile.mkdtemp(prefix=prefix)) / "gate"
    command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
               "-fsanitize=undefined", "-pthread", str(source),
               f"-I{ROOT}", "-o", str(binary)]
    try:
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True)
    finally:
        shutil.rmtree(binary.parent, ignore_errors=True)
    return " ".join(command[:-1] + ["<binary>"])


def public_kernel_gate() -> dict[str, object]:
    """The local secret-by-public kernels, on cases with known ring results."""
    return {
        "status": "passed",
        "source": PUBLIC_KERNEL_TEST.name,
        "sha256": file_digest(PUBLIC_KERNEL_TEST),
        "command": _compile_and_run(PUBLIC_KERNEL_TEST, "ghostmpc-kernel-gate-"),
    }


def round_counter_gate() -> dict[str, object]:
    """The round counter, on microprotocols of known round complexity.

    The counter the measured binary runs is `SerialRoundCounter` in
    `causal_round_counter.h`, which the patched `CommunicationObject`
    delegates to, so this drives the same code and cross-checks it against the
    declarative causal DAG in the same header.  It is a build gate rather than
    a test the runner trusts on faith: a build whose gate did not run must not
    publish rounds.
    """
    command = _compile_and_run(ROUND_COUNTER_TEST, "ghostmpc-round-gate-")
    return {
        "status": "passed",
        "source": ROUND_COUNTER_TEST.name,
        "sha256": file_digest(ROUND_COUNTER_TEST),
        "counter": "ghostmpc_porthos::SerialRoundCounter",
        "rule": "a round opens on the first receive after a non-receive; "
                "consecutive receives are one round; the PARALLEL_COMM "
                "exchanges and synchronize() announce themselves",
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path,
                        default=REPO / "third_party/EzPC-porthos-cache.git",
                        help="bare git cache holding the pinned EzPC commit; "
                             "created and fetched into if absent")
    parser.add_argument("--output", type=Path, required=True,
                        help="the disposable tree to create; must not exist")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="where to write porthos-build-manifest.json")
    parser.add_argument(
        "--threads-per-party", type=int, default=1,
        help="the declared compute budget this build is measured under. "
             "Porthos exposes no protocol-level thread count -- its MPC layer "
             "is single-threaded by construction -- so the budget is enforced "
             "at run time through EIGEN_NUM_THREADS/OMP_NUM_THREADS and is "
             "recorded here to say which knob carried it (default %(default)s)")
    parser.add_argument("--eigen-cache", type=Path,
                        default=REPO / "third_party/eigen-3.3.7.tar.gz",
                        help="Eigen 3.3.7 archive; downloaded and hash-checked "
                             "if absent")
    args = parser.parse_args()
    ensure_cache(args.cache)
    materialize(args.cache, args.output)
    prepare_eigen(args.eigen_cache, args.output)
    patch_athos(args.output)
    patch_porthos(args.output)
    round_counter = round_counter_gate()
    public_kernels = public_kernel_gate()
    integration_files = sorted(
        path for path in ROOT.iterdir()
        if path.is_file() and path.name not in {"__pycache__"})
    manifest = {
        "schema_version": 1, "upstream": LOCK,
        "disposable_tree": str(args.output.resolve()),
        "cache": str(args.cache.resolve()), "cache_modified": False,
        "eigen": {"url": EIGEN_URL, "version": "3.3.7",
                  "archive": str(args.eigen_cache.resolve()),
                  "sha256": EIGEN_SHA256},
        "patches": {path.name: file_digest(path) for path in integration_files},
        "gates": {"round_counter": round_counter,
                  "public_kernels": public_kernels},
        "patched_tree": {
            str(path.relative_to(args.output)): file_digest(path)
            for path in sorted((args.output / "Porthos/src").glob("ghostmpc*"))
        },
        "toolchain": {
            "platform": platform.platform(), "python": platform.python_version(),
            "cxx": compiler_version([os.environ.get("CXX", "c++"), "--version"]),
            "cmake": compiler_version(["cmake", "--version"]),
            "openssl": compiler_version(["openssl", "version"]),
        },
        "effective_contract": {
            "ring_bits": 64, "fractional_bits": 16,
            "threads_per_party": args.threads_per_party,
            "thread_knob": "EIGEN_NUM_THREADS/OMP_NUM_THREADS; Porthos has no "
                           "protocol-level thread count",
            "aes_ni_required": True, "public_weights": True,
            "aes_ni_observed": " aes " in (
                " " + Path("/proc/cpuinfo").read_text().lower() + " ")
                if Path("/proc/cpuinfo").is_file() else None,
            "external_additive_input": True, "online_output_reveal": False,
            "session_keys": "fresh per pool; supplied via argv[3]",
        },
        "reproduce": {
            "build": "baselines/porthos/build.sh",
            "prepare": "python baselines/porthos/prepare_build.py "
                       "--output <empty-dir> --manifest <manifest.json>",
            "round_counter_test": round_counter["command"],
            "public_kernel_test": public_kernels["command"],
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output / "porthos.lock.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
