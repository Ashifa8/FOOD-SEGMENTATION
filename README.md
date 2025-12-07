This project introduces a new idea where I use two types of co-occurrence matrices — a Spatial Co-Occurrence Matrix and a Dynamic Co-Occurrence Matrix — to improve segmentation results. The goal is to help the model understand rare or less-frequent classes that usually get ignored by traditional segmentation networks.

The Spatial Co-Occurrence Matrix captures stable relationships between classes that usually appear together in fixed spatial positions. The Dynamic Co-Occurrence Matrix captures relationships that change from image to image. After calculating both, I combine them to form a Fusion Matrix, which contains richer information. This fusion is then provided to a UNet model with a MobileViT encoder, allowing the model to learn better contextual features and recognize rare co-occurrence patterns more accurately.

I also used an auto.py script, which works as an automation file. It performs almost all steps automatically, such as:

loading the dataset

calculating spatial and dynamic co-occurrence matrices

generating the fusion matrix

training the UNet-MobileViT model

saving outputs and results

calculating evaluation metrics

The trained model is provided in .pth format, so anyone can easily load it and perform inference. The project also includes the fusion results, full source code, and supporting scripts needed to run everything smoothly.

To evaluate the performance of the co-occurrence-enhanced model, I calculated three metrics:

MSE (Mean Squared Error)

MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

These metrics help show how close the predicted outputs are to the ground truth and highlight the improvements gained by adding co-occurrence information into the segmentation pipeline.

