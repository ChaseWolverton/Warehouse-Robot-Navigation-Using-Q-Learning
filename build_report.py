from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from pathlib import Path
p=Path('Milestone2_Report.pdf')
doc=SimpleDocTemplate(str(p),pagesize=(612,792),rightMargin=49,leftMargin=49,topMargin=42,bottomMargin=40)
s=getSampleStyleSheet();s.add(ParagraphStyle(name='Main',parent=s['Normal'],fontSize=9.5,leading=13,spaceAfter=6));s.add(ParagraphStyle(name='Head',parent=s['Heading2'],fontSize=12,leading=15,spaceBefore=11,spaceAfter=5,textColor=colors.HexColor('#17345a')));s.add(ParagraphStyle(name='Title2',parent=s['Title'],fontSize=17,spaceAfter=7))
story=[]
def t(x,sty='Main'):story.append(Paragraph(x,s[sty]))
def fig(file,w,caption):
 from PIL import Image as PIL
 im=PIL.open(file); h=w*im.height/im.width
 story.append(Image(file,width=w,height=h));t(caption)
t('Warehouse Robot Navigation Using Q-Learning','Title2');t('ECE 414 | Final Project — Milestone 2: Environment Setup | November 2026')
t('1. Project Overview','Head')
t('• <b>Project idea:</b> Build a 6×6 warehouse where a robot picks up one package and delivers it while avoiding fixed and temporary obstacles and managing battery power.')
t('• <b>Implemented:</b> A Gymnasium-compatible Python environment with five actions, eight observation values, reward and termination rules, reproducible reset, a text renderer, and automated tests.')
t('• <b>Remaining:</b> Train a tabular Q-learning agent, implement random and rule-based baselines, compare results over multiple seeds, and test new obstacle layouts.')
t('2. Environment Design','Head')
t('<b>Environment:</b> A 6×6 grid with four fixed wall cells, two possible temporary obstacles, a robot, package, charger, and delivery location. Temporary obstacles are resampled each step and cannot block the robot or special locations.')
t('<b>State/observation:</b> [row, column, battery, carrying flag, package row, package column, temporary obstacle flag 1, temporary obstacle flag 2]. Battery is an integer from 0 to its configured maximum.')
t('<b>Actions:</b> 0=up, 1=down, 2=left, 3=right, 4=recharge. Pickup is automatic when the robot reaches the package. Recharging restores full battery only at the charging station.')
t('<b>Rewards:</b> −1 each step, another −2 for blocked movement or trying to recharge elsewhere, +30 upon delivery, and another −5 for running out of battery. Movement attempts use one battery unit, including blocked moves.')
t('<b>Transition:</b> An action changes position or battery; the package flag changes at pickup; temporary obstacles are resampled after nonterminal steps. Delivery and battery exhaustion terminate the episode; 100 steps truncate it.')
fig('figures/environment.png',340,'<b>Figure 1.</b> Warehouse visualization. Dark cells are walls; red cells are possible temporary blocks. The robot begins at the lower-left, and special cells are labeled.')
story.append(Spacer(1,8))
t('Environment implementation evidence','Head')
fig('figures/code_excerpt.png',485,'<b>Figure 2.</b> Excerpt from environment.py. The step() function applies movement, battery use, invalid-action penalties, pickup, and episode-ending rules.')
t('3. Environment Testing','Head')
t('The environment was checked with eight automated test cases for seeded reset, valid/invalid movement, charging, pickup and delivery, battery exhaustion, time-limit truncation, dynamic obstacles, and optional randomized package locations. The demonstration uses a fixed seed and prints each transition. Run the commands in README.md on Ubuntu to confirm behavior with the actual Gymnasium dependency.')
t('<b>Action behavior:</b> A valid movement changes the robot position and uses one battery unit. A blocked move leaves position unchanged but still costs energy and adds a penalty. At the package, carrying becomes true; at the delivery location, the episode ends successfully.')
t('<b>Reset:</b> Returns the start position, full battery, no package carried, zero steps, and seeded obstacle flags. Two calls with the same seed produce matching initial observations.')
t('<b>Technical challenges:</b> Battery is limited, and two cells may be temporarily blocked. Optional randomized pickup locations are also available for later generalization tests.')
# execution text excerpt
lines=Path('figures/sample_output.txt').read_text().splitlines()
subset=[x for x in lines if x.startswith('Step 04') or x.startswith('Step 10') or x.startswith('Delivery success')]
t('<b>Figure 3. Example demonstration output:</b>')
for x in subset:t('<font face="Courier" size="8">'+x.replace(' ','&#160;')+'</font>')
t('The demonstration reaches the pickup at step 4 and finishes the delivery at step 10 with 20 battery units remaining (using a demo battery capacity of 30). This example disables random obstacles to show a clear delivery path; the normal environment supports stochastic obstacles.')
t('4. Next Steps','Head')
t('Next I will implement epsilon-greedy tabular Q-learning and random/shortest-path baseline agents. I will compare delivery success, steps per successful delivery, remaining battery, and battery failures across multiple seeds and obstacle configurations.')
t('The main problems to investigate are Q-table size when battery and obstacle flags are included, finding a good reward balance so the agent completes deliveries instead of waiting, and making routes robust to blocked cells.')
def page(c,d):c.setFont('Helvetica',8);c.setFillColor(colors.grey);c.drawRightString(560,25,f'Page {d.page}')
doc.build(story,onFirstPage=page,onLaterPages=page)
print(p)
