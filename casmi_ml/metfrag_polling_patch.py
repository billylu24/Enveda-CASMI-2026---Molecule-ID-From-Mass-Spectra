"""Private runtime variant: only reduce the pinned candidate-pool wait constant."""

import hashlib
import shutil
import struct
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest

JAR_SHA = "8c55322fbc706c76df109dd886257262029695837bb1fb2211d4b88d945d8cfb"
CLASS_NAME = "de/ipbhalle/metfraglib/process/CombinedMetFragProcess.class"


def replace_wait_constant(data, expected_sha256):
    """Change one CONSTANT_Long from1000 to10; all other bytecode stays identical."""
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Pinned process class differs")
    if data[:4] != b"\xca\xfe\xba\xbe":
        raise ValueError("Java class required")
    count = struct.unpack_from(">H", data, 8)[0]
    index, offset, matches = 1, 10, []
    sizes = {
        3: 4,
        4: 4,
        5: 8,
        6: 8,
        7: 2,
        8: 2,
        9: 4,
        10: 4,
        11: 4,
        12: 4,
        15: 3,
        16: 2,
        17: 4,
        18: 4,
        19: 2,
        20: 2,
    }
    while index < count:
        tag = data[offset]
        offset += 1
        if tag == 1:
            length = struct.unpack_from(">H", data, offset)[0]
            offset += 2 + length
        elif tag in sizes:
            if tag == 5 and struct.unpack_from(">q", data, offset)[0] == 1000:
                matches.append(offset)
            offset += sizes[tag]
            if tag in (5, 6):
                index += 1
        else:
            raise ValueError("Unexpected Java constant tag")
        index += 1
    if len(matches) != 1:
        raise ValueError("Exactly one pinned1000ms wait constant required")
    at = matches[0]
    result = data[:at] + struct.pack(">q", 10) + data[at + 8 :]
    if len(data) != len(result):
        raise ValueError("Class size changed")
    return result


def prepare(jar, worker, output):
    jar, worker, output = Path(jar), Path(worker), Path(output)
    if digest(jar) != JAR_SHA:
        raise ValueError("Pinned2.6.11 jar required")
    if output.exists():
        raise ValueError("Fresh private classes directory required")
    with zipfile.ZipFile(jar) as archive:
        original = archive.read(CLASS_NAME)
    original_sha = hashlib.sha256(original).hexdigest()
    # The jar hash binds this exact class; no downloaded/recompiled algorithm source.
    patched = replace_wait_constant(original, original_sha)
    target = output / CLASS_NAME
    target.parent.mkdir(parents=True)
    target.write_bytes(patched)
    shutil.copy2(worker, output / "MetFragWorker.class")
    manifest = {
        "jar_sha256": JAR_SHA,
        "original_process_class_sha256": original_sha,
        "process_class_sha256": digest(target),
        "worker_class_sha256": digest(worker),
        "original_wait_ms": 1000,
        "wait_ms": 10,
        "changed_class_constant_count": 1,
        "scope": "Private derived MetFrag runtime class: only the candidate executor completion polling interval changes. All fragmentation/scoring bytecode, pinned jar, inputs and candidate order remain unchanged. Fresh exact-score verification required; no accuracy claim.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest
