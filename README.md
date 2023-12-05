# technical-assesment
technical assesment trimble

This repository is for the trimble assignment.

- report_kouassi_konan.pdf is the report of 1st deliverable

- Paper_sumarry.pdf is the article summary, you could find the real article here : https://arxiv.org/pdf/2303.11638.pdf

- inference.py is to test the model one or multiple image. You could run it using the command below:
    python script_name.py --model_path your_model.pth --image_path path_to_single_image --folder_path path_to_test_folder
    you'll find my best model here : https://drive.google.com/file/d/1A2zUHNdDQdU2b9jARwAUxRGUHaTOwXpP/view?usp=drive_link

- tuning.py is for the hyperparameter tuning. By runing it you'll have the best hyperparameters according to the differents hyperparameters you'll give

- main.py is the main function of this repo , by running it you could perform the training operation and get your best model
before training your dataset should be like this:


  train_set
  
      fields
          img.jpg
          ....
      roads
          img.jpg
          ....
  
  test_set
  
      fields
          img.jpg
          ....
      roads
          img.jpg
          ....



