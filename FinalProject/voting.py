import json
import sys
import os
import glob
from statistics import mode

if(__name__ == '__main__'):
    # read the json filepaths
    if(os.path.isdir(sys.argv[1])):
        filepaths = glob.glob(os.path.join(sys.argv[1], '*.json'))
    else:
        filepaths = []
        for i in range(1, len(sys.argv)):
            filepaths.append(sys.argv[i])

    # construct the vote dictionary
    vote_dict = {}
    ref_json = open(filepaths[0], 'r')
    ref_dict = json.load(ref_json)
    ref_json.close()
    for key in ref_dict.keys():
        vote_dict[key] = {}
        for i in range(len(ref_dict[key])):
            vote_dict[key][ref_dict[key][i]["question_id"]] = []
    
    # voting
    for filepath in filepaths:
        cur_json = open(filepath, 'r')
        cur_dict = json.load(cur_json)
        cur_json.close()
        for key in cur_dict.keys():
            for i in range(len(cur_dict[key])):
                q = cur_dict[key][i]["question_id"]
                a = cur_dict[key][i]["answer"]
                vote_dict[key][q].append(a)

    # calculate result
    result_dict = {}
    for key in vote_dict.keys():
        result_dict[key] = []
        for q in vote_dict[key].keys():
            vote_ans = mode(vote_dict[key][q])
            tmp_dict = {}
            tmp_dict["question_id"] = q
            tmp_dict["answer"] = vote_ans
            result_dict[key].append(tmp_dict)
    
    # export json file
    output_json = open("./voting_result.json", 'w')
    json.dump(result_dict, output_json, indent=2)
    output_json.close()
    print("Done")

