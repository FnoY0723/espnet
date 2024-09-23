'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-09-05 13:02:03
FilePath: /espnet/egs2/aishell/asr1/cal_latency.py
'''
import sys
import codecs
import glob
import os
import textgrid

def main():
    log_files = os.path.join(sys.argv[1], "asr_inference.1.log")
    start_times_marker = "INFO: " + "latency"
    end_times_marker = "INFO: " + "best hypo"
    text_grid_marker = "INFO: " + "text_grid_path"
    speech_len_marker = "INFO: " + "speech length"

    x = log_files
  
    all_latency = []
    first_latency = []
    last_latency = []
    correct_words = []
    lookahead=384

    first_ave = 0
    last_ave = 0
    count = 0

    with codecs.open(x, "r", "utf-8") as f:
        for line in f:
            x = line.strip()
            if text_grid_marker in x:
                text_grid_path = x.split(text_grid_marker + ": ")[1]
                utt_tg = textgrid.TextGrid.fromFile(text_grid_path)[0]
                start_time = utt_tg[1].maxTime*1000
                end_time = utt_tg[-2].maxTime*1000
                first_ave += start_time
                last_ave += end_time
                count += 1
            
            if speech_len_marker in x:
                speech_len = x.split(speech_len_marker + ": ")[1]
                speech_end_time = (float(speech_len)-1)//512 * 32

            if start_times_marker in x:
                w_ms = x.split(start_times_marker + ": ")[1]
                correct_words.append(w_ms.split(" ")[0])
                all_latency.append(float(w_ms.split(" ")[1].split("ms")[0]) + lookahead)

            elif end_times_marker in x:
                hyp = x.split(end_times_marker + ": ")[1]
                if len(correct_words)>0:
                    if correct_words[-1]==hyp[-1]:
                        if all_latency[-1]+ end_time + lookahead > speech_end_time:
                            all_latency[-1] = speech_end_time - end_time
                        last_latency.append(all_latency[-1])
                    if correct_words[0]==hyp[0]:
                        first_latency.append(all_latency[-len(correct_words)])
                correct_words = []
    

    first_latency.sort()
    mean_start_latency = sum(first_latency)/len(first_latency)
    print("mean start latency: ", mean_start_latency)
    print("median start latency: ", first_latency[int(len(first_latency)/2)])
    mean_start_latency_50 = sum(first_latency[:int(len(first_latency)*0.5)])/int(len(first_latency)*0.5)
    print("mean start latency of top 50%: ", mean_start_latency_50)
    mean_start_latency_90 = sum(first_latency[:int(len(first_latency)*0.9)])/int(len(first_latency)*0.9)
    print("mean start latency of top 90%: ", mean_start_latency_90)
    print("")
    last_latency.sort()
    mean_end_latency = sum(last_latency)/len(last_latency)
    print("mean end latency: ", mean_end_latency)
    print("median end latency: ", last_latency[int(len(last_latency)/2)])
    mean_end_latency_50 = sum(last_latency[:int(len(last_latency)*0.5)])/int(len(last_latency)*0.5)
    print("mean end latency of top 50%: ", mean_end_latency_50)
    mean_end_latency_90 = sum(last_latency[:int(len(last_latency)*0.9)])/int(len(last_latency)*0.9)
    print("mean end latency of top 90%: ", mean_end_latency_90)
    print("")
    all_latency.sort()
    mean_latency = sum(all_latency)/len(all_latency)
    print("mean latency: ", mean_latency)
    print("median latency: ", all_latency[int(len(all_latency)*0.5)])
    mean_latency_50 = sum(all_latency[:int(len(all_latency)*0.5)])/int(len(all_latency)*0.5)
    print("mean latency of top 50%: ", mean_latency_50)
    mean_latency_90 = sum(all_latency[:int(len(all_latency)*0.9)])/int(len(all_latency)*0.9)
    print("mean latency of top 90%: ", mean_latency_90)
    
    print("")
    print("first_ave: ", first_ave/count)
    print("last_ave: ", last_ave/count)

if __name__ == "__main__":
    main()