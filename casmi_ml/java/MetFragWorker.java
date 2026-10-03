import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;

/** Sequential process reuse only; all fragmentation/scoring remains in the pinned jar. */
public final class MetFragWorker {
    public static void main(String[] args) throws Exception {
        Method main = Class.forName("de.ipbhalle.metfrag.commandline.CommandLineTool")
            .getMethod("main", String[].class);
        BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
        for (String path; (path = input.readLine()) != null;) {
            if (path.isEmpty()) continue;
            main.invoke(null, (Object) new String[] {path});
            System.out.println("CASMI_METFRAG_DONE\t" + path);
            System.out.flush();
        }
    }
}
