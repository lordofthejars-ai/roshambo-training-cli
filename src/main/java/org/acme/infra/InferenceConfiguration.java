package org.acme.infra;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.OneHot;
import ai.djl.modality.cv.transform.RandomFlipLeftRight;
import ai.djl.modality.cv.transform.RandomFlipTopBottom;
import ai.djl.modality.cv.transform.RandomResizedCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.FixedPerVarTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Disposes;
import jakarta.enterprise.inject.Produces;
import java.io.IOException;
import java.nio.file.Paths;
import org.eclipse.microprofile.config.inject.ConfigProperty;

@ApplicationScoped
public class InferenceConfiguration {

    public static final int BATCH_SIZE = 32;
    public static final int NUMBER_OF_LABELS = 3;
    public static final String MODEL_NAME = "transferRoshambo";
    @ConfigProperty(name = "djl.resnet.model", defaultValue = "djl://ai.djl.pytorch/resnet18_embedding")
    String modelUrl;

    // ************************ Predict *************************

    @Produces
    @DjlZooModel
    @ApplicationScoped
    public ZooModel<NDList, NDList> createZooModel()
        throws ModelNotFoundException, MalformedModelException, IOException {
        Criteria<NDList, NDList> criteria = Criteria.builder()
            .setTypes(NDList.class, NDList.class)
            .optModelUrls(modelUrl)
            .optEngine("PyTorch")
            .optProgress(new ProgressBar())
            .optOption("trainParam", "false")
            .build();

        return criteria.loadModel();
    }

    @Produces
    @DjlModel
    @ApplicationScoped
    public Model createModel(@DjlZooModel ZooModel<NDList, NDList> embedding) throws MalformedModelException, IOException {

        Block baseBlock = embedding.getBlock();
        Block blocks =
            new SequentialBlock()
                .add(baseBlock)
                .addSingleton(nd -> nd.squeeze(new int[] {2, 3}))
                .add(Linear.builder().setUnits(NUMBER_OF_LABELS).build())
                .addSingleton(nd -> nd.softmax(1));

        Model model = Model.newInstance(MODEL_NAME);
        model.setBlock(blocks);
        model.load(Paths.get("model/"));

        return model;

    }

    @Produces
    @ApplicationScoped
    public Predictor<Image, Classifications>  createPredictor(@DjlModel Model model) {

        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
            .addTransform(new Resize(256))
            .addTransform(new CenterCrop(224, 224))
            .addTransform(new ToTensor())
            .addTransform(new Normalize(
                new float[] {0.485f, 0.456f, 0.406f},
                new float[] {0.229f, 0.224f, 0.225f}))

            .build();

        return model.newPredictor(translator);

    }

    // *********************** Training Configuration *******************

    @Produces
    @DjlTrainZooModel
    @ApplicationScoped
    public ZooModel<NDList, NDList> createZooModelForTraining()
        throws ModelNotFoundException, MalformedModelException, IOException {
        Criteria<NDList, NDList> criteria = Criteria.builder()
            .setTypes(NDList.class, NDList.class)
            .optModelUrls(modelUrl)
            .optEngine("PyTorch")
            .optProgress(new ProgressBar())
            .optOption("trainParam", "true")
            .build();

        return criteria.loadModel();
    }

    @Produces
    @DjlTrainModel
    @ApplicationScoped
    public Model createTrainModel(@DjlTrainZooModel ZooModel<NDList, NDList> embedding) throws ModelNotFoundException, MalformedModelException, IOException {

        Block baseBlock = embedding.getBlock();
        Block blocks =
            new SequentialBlock()
                .add(baseBlock)
                .addSingleton(nd -> nd.squeeze(new int[] {2, 3}))
                .add(Linear.builder().setUnits(NUMBER_OF_LABELS).build())
                .addSingleton(nd -> nd.softmax(1));

        Model model = Model.newInstance(MODEL_NAME);
        model.setBlock(blocks);

        return model;

    }

    @Produces
    @ApplicationScoped
    public Trainer createTrainer( @DjlTrainModel Model model) {
        // Configure trainer
        DefaultTrainingConfig config = setupTrainingConfig(model.getBlock());
        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());

        // Initialize the parameter shape and value

        Shape inputShape = new Shape(BATCH_SIZE, 3, 224, 224);
        trainer.initialize(inputShape);

        return trainer;
    }

    private static DefaultTrainingConfig setupTrainingConfig(Block baseBlock) {
        String outputDir = "target";
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
            trainer -> {
                TrainingResult result = trainer.getTrainingResult();
                Model model = trainer.getModel();
                float accuracy = result.getValidateEvaluation("Accuracy");
                model.setProperty("Accuracy", String.format("%.5f", accuracy));
                model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
            });

        DefaultTrainingConfig config =
            new DefaultTrainingConfig(new SoftmaxCrossEntropyLoss("SoftmaxCrossEntropy"))
                .addEvaluator(new Accuracy())
                .optDevices(Engine.getInstance().getDevices(1))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);

        // Customized learning rate
        float lr = 0.001f;
        FixedPerVarTracker.Builder learningRateTrackerBuilder =
            FixedPerVarTracker.builder().setDefaultValue(lr);
        for (Pair<String, Parameter> paramPair : baseBlock.getParameters()) {
            learningRateTrackerBuilder.put(paramPair.getValue().getId(), 0.1f * lr);
        }
        FixedPerVarTracker learningRateTracker = learningRateTrackerBuilder.build();
        Optimizer optimizer = Adam.builder().optLearningRateTracker(learningRateTracker).build();
        config.optOptimizer(optimizer);

        return config;
    }


    // ************* DataSets *****************

    @Produces
    @DjlTrainDatatset
    @ApplicationScoped
    public RandomAccessDataset createTrainDataset() throws TranslateException, IOException {
        return getData("train", BATCH_SIZE);
    }

    @Produces
    @DjlTestDataset
    @ApplicationScoped
    public RandomAccessDataset createTestDataset() throws TranslateException, IOException {
        return getData("test", BATCH_SIZE);
    }

    private static RandomAccessDataset getData(String usage, int batchSize)
        throws TranslateException, IOException {
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};

        // usage is either "train" or "test"
        Repository repository = Repository.newInstance("banana", Paths.get("./dataset/" + usage));

        System.out.printf("Using %s repository%n", repository.getBaseUri());

        ImageFolder dataset = ImageFolder.builder()
            .setRepository(repository)
            .addTransform(new RandomResizedCrop(256, 256)) // only in training
            .addTransform(new RandomFlipTopBottom()) // only in training
            .addTransform(new RandomFlipLeftRight()) // only in training
            .addTransform(new Resize(256, 256))
            .addTransform(new CenterCrop(224, 224))
            .addTransform(new ToTensor())
            .addTransform(new Normalize(mean, std))
            .addTargetTransform(new OneHot(NUMBER_OF_LABELS))
            .setSampling(batchSize, true)
            .build();
        dataset.prepare();
        return dataset;
    }

    // ****************** Cleaner *****************************

    void closeModel(@Disposes  @DjlTrainModel Model model) {
        model.close();
    }

    void closeEmbedding(@Disposes @DjlTrainZooModel ZooModel<NDList, NDList> embedding) {
        embedding.close();
    }

    void closePredictor(@Disposes Predictor<Image, Classifications> predictor) {
        predictor.close();
    }

}
