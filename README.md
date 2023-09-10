# Car Parts Segmentation
## _Using YOLOv7 and TorchServe_

This project leverages YOLOv7 for car parts detection and segmentation, and TorchServe for serving the model as a production-ready API. The goal of this project is to provide accurate and efficient car parts segmentation in images or video frames, making it suitable for various automotive applications.

## ✨Features✨

- Precise detection and segmentation of car parts, including doors, windows, tires, and more.
- Seamless integration with TorchServe for model deployment.
- Easy-to-use API for incorporating car parts segmentation into your applications.
- Real-time segmentation capability for video frames.
- Provides bounding boxes and masks for each detected car part.
- Supports various image and video formats as input.
- High-speed processing with GPU support.
- Detailed documentation for API usage.

## Tech Stack

Car Parts Segmentation is built on top of several open-source technologies:

- [YOLOv7] - A state-of-the-art object detection and segmentation model.
- [TorchServe] - A lightweight model serving framework for PyTorch models.
- [Python] - The programming language used for scripting and development.
- [OpenCV] - Open-source computer vision and machine learning library.

## Usage
To run Car Parts Segmentation, you'll need [Python](https://www.python.org/) 3.6+ installed on your system.

- Clone the repository:
    ```sh
    git clone https://github.com/yashjain-99/car_parts_segmentation.git
    ```
- Navigate to the project directory:
    ```sh
    cd car_parts_segmentation
    ```
- Install the required dependencies:
    ```py
    pip install -r requirements.txt
    ```
- Download the pre-trained YOLOv7 weights and place them in the `models_weights` folder.
- Start TorchServe to serve the model as an API:
    ```sh
    torchserve --start
    ```
- Use the API to perform car parts segmentation on an image:
    ```sh
    curl -X POST http://localhost:8080/predictions/car_parts_segmentation -T path/to/input_image.jpg -o path/to/output_image.jpg
    ```
    Replace `path/to/input_image.jpg` with the path to your input image and `path/to/output_image.jpg` with the desired output path.

- For real-time segmentation in a Python script:
    ```py
    from car_parts_segmentation import CarPartsSegmentation

    segmentation_model = CarPartsSegmentation(model_url="http://localhost:8080/predictions/car_parts_segmentation")
    result_image = segmentation_model.segment_image("path/to/input_image.jpg")
    result_image.save("path/to/output_image.jpg")
    ```

- Make sure to stop TorchServe when you're done:
    ```sh
    torchserve --stop
    ```

For detailed API documentation and usage examples, refer to the [API Documentation](api_docs.md).

**Feel free to reach out to me at your@email.com for any questions or suggestions!**

[YOLOv7]: <https://github.com/WongKinYiu/yolov7>
[TorchServe]: <https://pytorch.org/serve/>
[Python]: <https://www.python.org/>
[OpenCV]: <https://opencv.org/>
