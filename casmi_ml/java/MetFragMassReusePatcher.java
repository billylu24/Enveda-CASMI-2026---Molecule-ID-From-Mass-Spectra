import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.zip.*;
import org.objectweb.asm.*;
import org.objectweb.asm.tree.*;
import org.objectweb.asm.util.*;
import java.io.*;

/** Private derivative: same mass sum, reused only within one fragment's peak loop. */
public final class MetFragMassReusePatcher implements Opcodes {
    static final String PREFIX = "de/ipbhalle/metfraglib/";
    static final String FRAGMENT = PREFIX + "fragment/DefaultBitArrayFragment";
    static final String SCORER = PREFIX + "fragmenterassignerscorer/TopDownFragmenterAssignerScorer";
    static final String MASS = "(L" + PREFIX + "interfaces/IMolecularStructure;)D";
    static String sha(byte[] bytes) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }
    static ClassNode read(byte[] bytes) {
        ClassNode node = new ClassNode(); new ClassReader(bytes).accept(node, 0); return node;
    }
    static byte[] encode(ClassNode node) {
        ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
        node.accept(writer); return writer.toByteArray();
    }
    static AbstractInsnNode realPrevious(AbstractInsnNode n) {
        do { n = n.getPrevious(); } while (n != null && n.getOpcode() < 0); return n;
    }
    static Map<String, String> methods(byte[] bytes) {
        ClassNode node = new ClassNode();
        new ClassReader(bytes).accept(node, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
        Map<String, String> result = new TreeMap<>();
        for (MethodNode m : node.methods) {
            Textifier printer = new Textifier();
            m.accept(new TraceMethodVisitor(printer));
            StringWriter text = new StringWriter(); printer.print(new PrintWriter(text));
            result.put(m.name + m.desc, text.toString());
        }
        return result;
    }
    static void verifyUnchanged(byte[] original, byte[] derived, String allowed) {
        Map<String, String> a = methods(original), b = methods(derived);
        for (String key : a.keySet())
            if (!key.startsWith(allowed + "(") && !a.get(key).equals(b.get(key)))
                throw new IllegalStateException("Unchanged original method differs: " + key);
    }
    public static void main(String[] args) throws Exception {
        byte[] fragmentBytes, scorerBytes;
        try (ZipFile jar = new ZipFile(args[0])) {
            fragmentBytes = jar.getInputStream(jar.getEntry(FRAGMENT + ".class")).readAllBytes();
            scorerBytes = jar.getInputStream(jar.getEntry(SCORER + ".class")).readAllBytes();
        }
        if (!sha(fragmentBytes).equals("02b5d62b16a80f3355f6d841d5669aa9eb5190be33360ded156d9a86ad01e86b") ||
            !sha(scorerBytes).equals("27172ecc68adf42547be538ea7f6105c114dc1fbd964f278b176ca0f100387fc"))
            throw new IllegalStateException("Pinned original classes differ");
        ClassNode fragment = read(fragmentBytes), scorer = read(scorerBytes);
        MethodNode original = fragment.methods.stream().filter(m -> m.name.equals("matchToPeak")).findFirst().orElseThrow();
        MethodNode reused = new MethodNode(ASM9, original.access, "casmiMatchToPeakWithMass", original.desc.replace(")", "D)"), null, null);
        original.accept(reused);
        int massCalls = 0;
        for (AbstractInsnNode n : reused.instructions.toArray()) {
            if (n instanceof MethodInsnNode && ((MethodInsnNode)n).name.equals("getMonoisotopicMass")) {
                MethodInsnNode call = (MethodInsnNode)n;
                if (!call.owner.equals(FRAGMENT) || !call.desc.equals(MASS)) throw new IllegalStateException("Unexpected mass call");
                AbstractInsnNode precursor = realPrevious(n), receiver = realPrevious(precursor);
                if (!(precursor instanceof VarInsnNode) || precursor.getOpcode() != ALOAD || ((VarInsnNode)precursor).var != 1 ||
                    !(receiver instanceof VarInsnNode) || receiver.getOpcode() != ALOAD || ((VarInsnNode)receiver).var != 0)
                    throw new IllegalStateException("Unexpected mass operands");
                reused.instructions.remove(receiver); reused.instructions.remove(precursor);
                reused.instructions.set(n, new VarInsnNode(DLOAD, 6)); massCalls++;
            }
        }
        if (massCalls != 1) throw new IllegalStateException("One original mass call required");
        // Original locals start at6; new double parameter requires shifting all old locals.
        for (AbstractInsnNode n : reused.instructions.toArray()) {
            if (n instanceof VarInsnNode && ((VarInsnNode)n).var >= 6 && n.getOpcode() != DLOAD)
                ((VarInsnNode)n).var += 2;
            else if (n instanceof VarInsnNode && ((VarInsnNode)n).var >= 6 && n.getOpcode() == DLOAD && ((VarInsnNode)n).var != 6)
                ((VarInsnNode)n).var += 2;
            if (n instanceof IincInsnNode && ((IincInsnNode)n).var >= 6) ((IincInsnNode)n).var += 2;
            if (n instanceof FrameNode) reused.instructions.remove(n);
        }
        reused.localVariables = null;
        fragment.methods.add(reused);
        MethodNode calculate = scorer.methods.stream().filter(m -> m.name.equals("calculate") && m.desc.equals("()V")).findFirst().orElseThrow();
        int slot = calculate.maxLocals, changes = 0;
        for (AbstractInsnNode n : calculate.instructions.toArray()) {
            if (n instanceof MethodInsnNode && ((MethodInsnNode)n).name.equals("matchToPeak")) {
                MethodInsnNode call = (MethodInsnNode)n;
                if (!call.owner.equals(PREFIX + "fragment/AbstractTopDownBitArrayFragment") || !call.desc.equals(original.desc))
                    throw new IllegalStateException("Unexpected matching call");
                AbstractInsnNode anchor = n;
                while (anchor != null && !(anchor.getOpcode() == ICONST_M1 && realPrevious(anchor) instanceof JumpInsnNode)) anchor = anchor.getPrevious();
                if (anchor == null || !(anchor.getNext() instanceof VarInsnNode) || ((VarInsnNode)anchor.getNext()).var != 19)
                    throw new IllegalStateException("Pinned fragment-loop entry not found");
                InsnList load = new InsnList();
                LabelNode skip = new LabelNode(), done = new LabelNode();
                load.add(new VarInsnNode(ILOAD, 16)); load.add(new JumpInsnNode(IFLT, skip));
                load.add(new VarInsnNode(ALOAD, 18)); load.add(new VarInsnNode(ALOAD, 1));
                load.add(new MethodInsnNode(INVOKEVIRTUAL, PREFIX + "fragment/AbstractTopDownBitArrayFragment", "getMonoisotopicMass", MASS, false));
                load.add(new VarInsnNode(DSTORE, slot)); load.add(new JumpInsnNode(GOTO, done));
                load.add(skip); load.add(new InsnNode(DCONST_0)); load.add(new VarInsnNode(DSTORE, slot)); load.add(done);
                calculate.instructions.insertBefore(anchor, load);
                calculate.instructions.insertBefore(n, new VarInsnNode(DLOAD, slot));
                call.name = reused.name; call.desc = reused.desc; changes++;
            }
        }
        if (changes != 1) throw new IllegalStateException("One pinned peak traversal required");
        for (AbstractInsnNode n : calculate.instructions.toArray()) if (n instanceof FrameNode) calculate.instructions.remove(n);
        Path root = Path.of(args[1]);
        byte[] patchedFragment = encode(fragment), patchedScorer = encode(scorer);
        verifyUnchanged(fragmentBytes, patchedFragment, "");
        verifyUnchanged(scorerBytes, patchedScorer, "calculate");
        for (ClassNode node : List.of(fragment, scorer)) {
            Path target = root.resolve(node.name + ".class"); Files.createDirectories(target.getParent()); Files.write(target, encode(node));
        }
        System.out.println("original_fragment_sha256=" + sha(fragmentBytes));
        System.out.println("original_scorer_sha256=" + sha(scorerBytes));
        System.out.println("changes=one_added_match_overload/one_mass_reuse_peak_loop; original_match_and_mass_methods_retained");
    }
}
