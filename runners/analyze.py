import re
from collections import defaultdict
import glob
def sort_live_range_report(fname):
    data = defaultdict(list)
    totals = {}
    cur_part = None
    with open(fname, 'r') as f:
        lines = f.read()
        lines = lines.split("\n")
        for line in lines:
            if "Part[" in line:
                lparts = line.split()
                matches = re.match("Part\\[\\d+\\]", lparts[0])
                # print(matches)
                part_id = matches[0].split('[')[1].split(']')[0]
                cur_part = part_id
                totals[cur_part] = lparts[-1]
            elif line and cur_part is not None:
                line = line.strip()
                parts = line.split()
                data[cur_part].append(parts)
            elif cur_part is not None:
                break
    print({k: float(v)/1e9 for k, v in totals.items()})
    # print([int(x)/1e9 for x in totals.values()], 'GB')
    print(sum([int(x) for x in totals.values()])//1e9, 'GB')
    max_part = None
    max_t = 0
    print('Finding max partition')
    for p,t in totals.items():
        if int(t) > max_t:
            max_part = p
            max_t = int(t)
            print('  Updating max to', max_part, max_t, max_t/1e9, 'GB')
    print('Printing tensors from max partition', max_part)
    sorted_part_tensors = sorted(data[max_part], key=lambda x: int(x[-1]), reverse=True)
    for lt in sorted_part_tensors:
        if float(lt[2])/1e6 > 50:
            print(' ', lt[0], lt[1], int(lt[2])/1e6, 'MB')



if __name__  == "__main__":
    import sys

    # if len(sys.argv) == 1:
    #     # sort_live_range_report("/fsx/huilgolr/axlearn/artifacts/11382/neuron_dump/pid85232-program0/LiveRangeReport_PostHloPart.txt")
    #     # sort_live_range_report("/fsx/huilgolr/axlearn/artifacts/11149/neuron_dump/pid862438-program0/LiveRangeReport_PostHloPart.txt")
    #     sort_live_range_report("/fsx/huilgolr/axlearn/artifacts/11292/neuron_dump/pid674413-program0/LiveRangeReport_PostHloPart.txt")
    if len(sys.argv) == 2:
        job_id = sys.argv[1]
        for fpath in glob.glob(f"/fsx/huilgolr/axlearn/artifacts/{job_id}/neuron_dump/pid*-program*/LiveRangeReport_PostHloPart.txt"):
            print(fpath)
            sort_live_range_report(fpath)
    else:
        raise NotImplementedError