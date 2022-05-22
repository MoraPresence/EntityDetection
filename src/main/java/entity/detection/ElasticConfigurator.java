package entity.detection;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.typesafe.config.Config;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.TransportAddress;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.aggregations.AggregationBuilders;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;
import org.elasticsearch.search.aggregations.bucket.terms.TermsAggregationBuilder;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.transport.client.PreBuiltTransportClient;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

public class ElasticConfigurator {
    private Config config;
    private PreBuiltTransportClient client;

    private final static String INDEX_NAME = "news";


    public static class OneNews {
        private String _header;
        private String _author;
        private String _text;
        private String _URI;

        public String getHeader() {
            return _header;
        }

        public String getAuthor() {
            return _author;
        }

        public String getText() {
            return _text;
        }

        public String getURI() {
            return _URI;
        }

        public void setHeader(String header) {
            this._header = header;
        }

        public void setAuthor(String author) {
            this._author = author;
        }

        public void setText(String text) {
            this._text = text;
        }

        public void setURI(String URI) {
            this._URI = URI;
        }

    }

    private PreBuiltTransportClient createClient() throws UnknownHostException {
        Settings settings = Settings.builder()
                .put("cluster.name", config.getString("cluster"))
                .build();

        PreBuiltTransportClient cli = new PreBuiltTransportClient(settings);

        cli.addTransportAddress(
                new TransportAddress(InetAddress.getByName(config.getString("host")), config.getInt("port"))
        );
        return cli;
    }

    //Fixme: unsafe! check for existing client or close app!
    public void initialize(Config conf) {
        config = conf;
        try {
            client = createClient();
        } catch (UnknownHostException e) {
            e.printStackTrace();
        }
    }

    void getSomeDataAll() {
        QueryBuilder query = QueryBuilders.matchAllQuery();
        SearchResponse response = client.prepareSearch("*").setQuery(query).get();
        // System.out.println(response.getHits().getTotalHits());
    }

    public List<OneNews> search(String key, String searchString) throws ExecutionException, InterruptedException {
        SearchRequest searchRequest = new SearchRequest(INDEX_NAME);
        SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
        if (key.isEmpty() || searchString.isEmpty()) searchSourceBuilder.query(QueryBuilders.matchAllQuery());
        else searchSourceBuilder.query(QueryBuilders.matchQuery(key, searchString));

        searchRequest.source(searchSourceBuilder);
        SearchResponse searchResponse = client.search(searchRequest).get();
        List<OneNews> news = new ArrayList<>();
        for (SearchHit hit : searchResponse.getHits().getHits()) {
            Map<String, Object> sourceAsMap = hit.getSourceAsMap();
            OneNews oneNews = new OneNews();
            oneNews.setHeader((String) sourceAsMap.get("header"));
            oneNews.setAuthor((String) sourceAsMap.get("author"));
            oneNews.setText((String) sourceAsMap.get("text"));
            oneNews.setURI((String) sourceAsMap.get("uri"));
            news.add(oneNews);
        }
        return news;
    }

    public void aggregation() throws ExecutionException, InterruptedException {
        SearchRequest searchRequest = new SearchRequest(INDEX_NAME);
        SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
        TermsAggregationBuilder aggregationBuilder = AggregationBuilders.terms("author_count").field("author.keyword");

        searchSourceBuilder.aggregation(aggregationBuilder);
        searchRequest.source(searchSourceBuilder);
        SearchResponse searchResponse = client.search(searchRequest).get();
        Terms terms = searchResponse.getAggregations().get("author_count");

        for (Terms.Bucket bucket : terms.getBuckets()) {
            System.out.println("author=" + bucket.getKey() + " count=" + bucket.getDocCount());
        }
    }
}
