package org.acme;

import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.OneHot;
import ai.djl.modality.cv.transform.RandomFlipLeftRight;
import ai.djl.modality.cv.transform.RandomFlipTopBottom;
import ai.djl.modality.cv.transform.RandomResizedCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.repository.Repository;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.translate.TranslateException;
import jakarta.inject.Inject;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.acme.infra.DjlTestDataset;
import org.acme.infra.DjlTrainDatatset;

import org.acme.infra.InferenceConfiguration;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

@Command(name = "greeting", mixinStandardHelpOptions = true)
public class ImageClassificationCommand implements Runnable {

    @Option(names = {"-s", "synset"}, description = "create the synset archive")
    boolean synset;

    @Option(names = {"-t", "train"}, description = "train the model")
    boolean train;

    @Option(names = {"-p", "predict"}, description = "predict an image")
    boolean predict;

    @Parameters(arity = "0..1", paramLabel = "FILE", description = "image to categorize")
    Path image;

    @Inject
    Trainer trainer;

    @Inject
    Predictor<Image, Classifications> predictor;

    @Inject
    @DjlTrainDatatset
    RandomAccessDataset datasetTrain;

    @Inject
    @DjlTestDataset
    RandomAccessDataset datasetTest;

    @Override
    public void run() {

        if (synset) {
            try {
                createSynset();
            } catch (TranslateException | IOException e) {
                throw new RuntimeException(e);
            }

            System.exit(0);
        }

        if (train) {
            try {
                train();
            } catch (IOException | TranslateException e) {
                throw new RuntimeException(e);
            }

            System.exit(0);
        }

        if (predict) {
            try {
                System.out.println(image);
                Image img = ImageFactory.getInstance().fromFile(image);
                System.out.println(predictor.predict(img).getAsString());
            } catch (IOException | TranslateException e) {
                throw new RuntimeException(e);
            }

            System.exit(0);
        }

    }

    public String train() throws IOException, TranslateException {

        // Train
        EasyTrain.fit(trainer, 10, datasetTrain, datasetTest);

        // Save model
        trainer.getModel().save(Paths.get("./model/"), InferenceConfiguration.MODEL_NAME);

        TrainingResult trainingResult = trainer.getTrainingResult();

        return trainingResult.toString();
    }

    private void createSynset() throws TranslateException, IOException {

        Repository repository = Repository.newInstance("banana", Paths.get("./dataset/train"));
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        int batchSize = 32;

        ImageFolder dataset = ImageFolder.builder()
            .setRepository(repository)
            .addTransform(new RandomResizedCrop(256, 256)) // only in training
            .addTransform(new RandomFlipTopBottom()) // only in training
            .addTransform(new RandomFlipLeftRight()) // only in training
            .addTransform(new Resize(256, 256))
            .addTransform(new CenterCrop(224, 224))
            .addTransform(new ToTensor())
            .addTransform(new Normalize(mean, std))
            .addTargetTransform(new OneHot(3))
            .setSampling(batchSize, true)
            .build();

        Files.write(Paths.get("model", "synset.txt"), dataset.getSynset());

    }

}
