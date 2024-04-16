# Cricket Analysis

The above project is made in OpenCV using Media Pipe and other technologies. In order to make use of the project, follow the given steps:

1) Clone the given current repo:

```bash
git clone https://github.com/Harri200191/AI_Enabled_Gym.git
```

2) Install the required dependencies using:

```bash
pip install -r requirements.txt
```

3) Use the analyzer by creating a new python file and using it as:

```python
# Add path of your video and specify left arm or right arm
from utils import *
Analyzer = BowlingActionAnalyser(video = 'Path of your video', bowler='right arm')
Analyzer.analyze()
```

4) If you want to use your own camera and perform the analysis real time, do

```python
from utils import *
Analyzer = BowlingActionAnalyser(video = 'Path of your video', bowler='right arm', position=0)
Analyzer.analyze()
```

5) Follow the example usage given in the main.ipynb file to run the project

I hope you enjoy using the tool!