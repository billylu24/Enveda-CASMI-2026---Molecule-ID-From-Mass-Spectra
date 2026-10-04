import java.lang.reflect.*;
import java.util.*;
import de.ipbhalle.metfraglib.fragment.DefaultBitArrayFragment;
import de.ipbhalle.metfraglib.interfaces.*;
import de.ipbhalle.metfraglib.match.FragmentMassToPeakMatch;
import de.ipbhalle.metfraglib.parameter.Constants;
import de.ipbhalle.metfraglib.peak.TandemMassPeak;

/** Edge cases for the private overload, using the retained original as comparator. */
public final class MetFragMassReuseCheck {
    static final class Fragment extends DefaultBitArrayFragment {
        Fragment(IMolecularStructure molecule, int depth, int hydrogens) {
            super(molecule); this.treeDepth = (byte)depth; setNumberHydrogens(hydrogens);
        }
    }
    static void require(boolean value) { if (!value) throw new AssertionError("Match behavior differs"); }
    public static void main(String[] args) throws Exception {
        Random rng = new Random(20261004);
        double[] masses = new double[12];
        IMolecularStructure precursor = (IMolecularStructure)Proxy.newProxyInstance(
            MetFragMassReuseCheck.class.getClassLoader(), new Class<?>[] {IMolecularStructure.class},
            (proxy, method, values) -> {
                switch (method.getName()) {
                    case "getNonHydrogenAtomCount": return masses.length;
                    case "getNonHydrogenBondCount": return masses.length - 1;
                    case "getMassOfAtom": return masses[(Integer)values[0]];
                    default: throw new UnsupportedOperationException(method.getName());
                }
            });
        int checks = 0;
        for (int trial = 0; trial < 100; trial++) {
            for (int i = 0; i < masses.length; i++) masses[i] = 12 + rng.nextDouble() * 20;
            int depth = trial % 4, hydrogens = trial % 9;
            Fragment original = new Fragment(precursor, depth, hydrogens), reused = new Fragment(precursor, depth, hydrogens);
            for (int index : new int[] {0, 1, 2, 3, 4, 5, 7, 8, 9}) {
                for (boolean positive : new boolean[] {true, false}) {
                    for (int shift = -3; shift <= 3; shift++) {
                        for (double offset : new double[] {-0.1, -0.00200001, -0.002, -0.0001, 0, 0.0001, 0.002, 0.00200001, 0.1}) {
                            // Change atom mask and precursor between calls: no persisted cache is allowed.
                            int atom = rng.nextInt(masses.length); boolean selected = rng.nextBoolean();
                            original.setAtomBit(atom, selected); reused.setAtomBit(atom, selected);
                            masses[atom] = 12 + rng.nextDouble() * 20;
                            double mass = original.getMonoisotopicMass(precursor);
                            require(Double.doubleToRawLongBits(mass) == Double.doubleToRawLongBits(reused.getMonoisotopicMass(precursor)));
                            double peakMass = mass + Constants.getIonisationTypeMassCorrection(index, positive) + shift * Constants.HYDROGEN_MASS + offset;
                            TandemMassPeak peak = new TandemMassPeak(peakMass, 100);
                            peak.setMassLimitsByMassDeviations(0.002, 10);
                            IMatch[] a = new IMatch[1], b = new IMatch[1];
                            byte left = original.matchToPeak(precursor, peak, index, positive, a);
                            byte right = reused.casmiMatchToPeakWithMass(precursor, peak, index, positive, b, mass);
                            require(left == right && (a[0] == null) == (b[0] == null));
                            if (a[0] != null) {
                                require(a[0].getMatchedFragmentsSize() == b[0].getMatchedFragmentsSize());
                                require(a[0].isPositiveCharge() == b[0].isPositiveCharge());
                                for (int i = 0; i < a[0].getMatchedFragmentsSize(); i++) {
                                    require(a[0].getNumberOfOverallHydrogensDifferToPeakMass(i) == b[0].getNumberOfOverallHydrogensDifferToPeakMass(i));
                                    double am = ((FragmentMassToPeakMatch)a[0]).getMatchedFragmentMassToTandemMassPeak(i);
                                    double bm = ((FragmentMassToPeakMatch)b[0]).getMatchedFragmentMassToTandemMassPeak(i);
                                    require(Double.doubleToRawLongBits(am) == Double.doubleToRawLongBits(bm));
                                }
                            }
                            checks++;
                        }
                    }
                }
            }
            require(original.matchToPeak(precursor, new TandemMassPeak(100, 10), 1, true, null) == reused.casmiMatchToPeakWithMass(precursor, new TandemMassPeak(100, 10), 1, true, null, 0));
        }
        System.out.println("Exact original/overload peak comparisons=" + checks + "; changed precursor/mask, ion signs, hydrogen shifts, boundary masses and null outputs checked");
    }
}
