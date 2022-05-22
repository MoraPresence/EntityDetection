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

@Slf4j
public class entityDetection {
    private final static String DEFAULT_PATH_PARSER_MODEL =
            "src/main/resources/nndep.rus.model.wiki.txt.gz";
    private final static String DEFAULT_PATH_TAGGER =
            "src/main/resources/russian-ud-pos.tagger";
    private final static String DEFAULT_PATH_MF_TAGGER =
            "src/main/resources/russian-ud-mf.tagger";
    private final static String DEFAULT_LEMMA_DICT = "src/main/resources/dict.tsv";
    //private final static String DEFAULT_PATH_TEXT = "ru_example.txt";
    private final static boolean MF = true;

    public static StanfordCoreNLP buildPipeline() {
        String tagger = DEFAULT_PATH_TAGGER;
        String taggerMF = DEFAULT_PATH_MF_TAGGER;
        String parser = DEFAULT_PATH_PARSER_MODEL;
        String pLemmaDict = DEFAULT_LEMMA_DICT;

        boolean mf = MF;


        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        if (mf) {
            pipeline.addAnnotator(new RussianMorphoAnnotator(new MaxentTagger(taggerMF)));
        }
        pipeline.addAnnotator(new POSTaggerAnnotator(new MaxentTagger(tagger)));

        Properties propsParser = new Properties();
        propsParser.setProperty("model", parser);
        propsParser.setProperty("tagger.model", tagger);
        pipeline.addAnnotator(new DependencyParseAnnotator(propsParser));

        if (pLemmaDict.isEmpty()) {
            pipeline.addAnnotator(new RussianLemmatizationAnnotator());
        } else {
            pipeline.addAnnotator(new RussianLemmatizationAnnotator(pLemmaDict));
        }
        return pipeline;
    }
}