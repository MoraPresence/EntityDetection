package entity.detection;

       /* Scanner sc = new Scanner(System.in,"utf-8");
        System.setProperty("console.encoding","utf-8");
        PrintStream ps = new PrintStream(System.out, true, "windows-1251");*/

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import lombok.extern.slf4j.Slf4j;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ExecutionException;

import static entity.detection.entityDetection.buildPipeline;

@Slf4j
public class Main {
    public static void main(String[] args) throws FileNotFoundException, IOException, ExecutionException, InterruptedException {

        CoreDocument docs = new CoreDocument("Однажды весною, в час небывало жаркого заката, в Москве, на Патриарших прудах, появились два гражданина.");
        StanfordCoreNLP pipeline = buildPipeline();

        pipeline.annotate(docs);
        Annotation annotation = docs.annotation();

        log.info("{}", annotation);
        List<CoreLabel> coreLabels = annotation.get(CoreAnnotations.TokensAnnotation.class);
        for (CoreLabel cl : coreLabels) {
            log.info("{} {} {}", cl.word(), cl.lemma(), cl.tag());
        }

        /*Config conf = ConfigFactory.load();
        ElasticConfigurator esCon = new ElasticConfigurator();
        esCon.initialize(conf.getConfig("es"));

        List<ElasticConfigurator.OneNews> testSearch2 = esCon.search("", "");
        System.out.println(testSearch2.get(0).getText());*/
    }
}
