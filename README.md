# DLCV Final Project ( STAR )

# How to run your code?
* TODO: Please provide the scripts for TAs to reproduce your results, including training and inference. For example, 
```shell script=
bash train.sh <Path to videos folder> <annotation file> [additional path(s)...]
bash inference.sh <Path to videos folder> <annotation file>
```
* For the **training code**, feel free to add extra paths to your training script if you want to use additional data such as bounding boxes or hyper-graphs.
* You can modify `.gitignore` file to avoid uploading your data

# Usage
To start working on this final project, you should clone this repository into your local machine by the following command:

    git clone https://github.com/ntudlcv/DLCV-Fall-2023-Final-1-<team name>.git
  
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1TsR0l84wWNNWH7HaV-FEPFudr3o9mVz29LZQhFO22Vk/edit?usp=sharing) to view the slides of Final Project - STAR Benchmark. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

# Dataset Overview
The following files are required for use in your training process.

### Question, Multiple Choice Answers and Situation Graphs

* Questions and Answers (.json) : [Train](https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/STAR_train.json) [Val](https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/STAR_val.json) [Test](https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/STAR_test.json)
* Split file (Optional): [Train/Val/Test Split File (.json)](https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/split_file.json)

### Video Data  
* [raw video data](https://prior.allenai.org/projects/charades): recommend: Data(scaled up to 480p)

If you want to use additional data such as bounding boxes, hyper-graphs, ..., please refer to the following links for more information.
* [Star Official Website](https://bobbywu.com/STAR/#repo)
* [GitHub](https://github.com/csbobby/STAR_Benchmark)


# Submission Rules
### Deadline
112/12/28 (Thur.) 23:59 (GMT+8)
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion
