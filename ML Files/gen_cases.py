import csv, random
R=random.Random(20261116)
ANY=["pregame","first_half","halftime","second_half","postgame"]
LIGHT={"pregame":["day"],"first_half":["day","dusk"],"halftime":["dusk"],"second_half":["night"],"postgame":["night"]}
CONSUMER=(4,20); DELIV=(14,28)
# id, group, name, label, response, params
S=[
(1,"A","Delivery drone on filed corridor outside restricted zone","authorized","log_only",dict(n=1,range=(5600,8000),alt=(60,120),speed=DELIV,path="transit",closest=(5600,7500),rf=True,rid="valid",listed=True,dev=(0,25),target="none")),
(2,"A","Delivery drone detouring around weather or traffic","authorized","log_only",dict(n=1,range=(5600,8000),alt=(60,120),speed=DELIV,path="transit_detour",closest=(5200,7000),rf=True,rid="valid",listed=True,dev=(40,250),target="none")),
(3,"A","Broadcast drone inside approved box over field","authorized","log_only",dict(n=1,range=(0,80),alt=(30,70),speed=(0,10),path="box_patrol",closest=(0,60),rf=True,rid="valid",listed=True,dev=(0,15),target="none",launch="inside_authorized")),
(4,"A","Police or security drone on patrol","authorized","log_only",dict(n=(1,2),range=(150,900),alt=(40,110),speed=(3,15),path="orbit",closest=(120,600),rf=True,rid="valid",listed=True,dev=(0,40),target="none")),
(5,"A","Halftime drone light show (approved)","authorized","log_only",dict(n=(100,500),range=(0,150),alt=(40,120),speed=(0,6),path="formation_show",closest=(0,80),rf=True,rid="valid",listed=True,dev=(0,10),target="none",phase=["halftime"],swarm=True,launch="inside_authorized")),
(6,"A","Approved campus research flight nearby","authorized","log_only",dict(n=1,range=(900,3500),alt=(20,100),speed=(2,12),path="survey_grid",closest=(800,3200),rf=True,rid="valid",listed=True,dev=(0,30),target="none")),
(7,"A","Crewed aircraft (medical helicopter or TV aircraft)","authorized","filter_out",dict(n=1,range=(3000,9000),alt=(150,600),speed=(35,75),path="transit",closest=(200,4000),rf=False,rid="adsb",listed=[True,False],dev=0,target="none",obj="crewed_aircraft")),
(8,"A","Pregame military flyover","authorized","filter_out",dict(n=(2,4),range=(8000,12000),alt=(300,500),speed=(150,250),path="flyover",closest=(0,150),rf=False,rid="none",listed=True,dev=0,target="none",phase=["pregame"],obj="military_jet")),
(9,"B","Bird flock crossing the stadium","non_drone","filter_out",dict(n=(5,200),range=(400,2500),alt=(10,150),speed=(8,18),path="wander_flock",closest=(0,600),obj="bird")),
(10,"B","Single large bird circling","non_drone","filter_out",dict(n=1,range=(100,1500),alt=(40,300),speed=(6,14),path="soar_circle",closest=(0,800),obj="bird")),
(11,"B","Birds or bats around the lights at night","non_drone","filter_out",dict(n=(3,60),range=(0,300),alt=(20,60),speed=(4,12),path="erratic",closest=(0,150),obj="bird_or_bat",phase=["second_half","postgame"])),
(12,"B","Balloons released by fans","non_drone","filter_out",dict(n=(1,300),range=(0,200),alt=(5,40),speed=(1,6),path="drift_up",closest=(0,100),obj="balloon",launch="inside")),
(13,"B","Fireworks or pyrotechnics","non_drone","filter_out",dict(n=(5,80),range=(0,250),alt=(20,150),speed=(20,60),path="ballistic",closest=(0,200),obj="pyrotechnic",phase=["pregame","halftime","postgame"])),
(14,"B","Kite or drifting debris","non_drone","filter_out",dict(n=1,range=(50,1200),alt=(5,80),speed=(0,8),path="drift",closest=(0,900),obj="debris")),
(15,"B","Phone hotspot or Wi-Fi device resembling a drone signal","non_drone","filter_out",dict(n=1,range=(0,400),alt=(0,40),speed=(0,2),path="static",closest=(0,400),obj="rf_only_ghost",rf=True)),
(16,"B","Insects close to a camera lens","non_drone","filter_out",dict(n=(1,20),range=(0,5),alt=(35,45),speed=(0,3),path="erratic",closest=(0,5),obj="camera_only_ghost")),
(17,"C","Hobbyist hovering over a lot, filming from outside","unauthorized_careless","dispatch_police_to_pilot",dict(n=1,range=(250,900),alt=(30,120),speed=(0,5),path="hover",closest=(250,900),rf=True,rid=["valid","none"],listed=False,target="none",launch="adjacent_lot")),
(18,"C","Hobbyist flying over the bowl for video","unauthorized_careless","alert_operator",dict(n=1,range=(300,1500),alt=(40,120),speed=CONSUMER,path="direct_then_hover",closest=(0,90),rf=True,rid=["valid","none"],listed=False,target="bowl")),
(19,"C","Toy drone flown low at a tailgate","unauthorized_careless","dispatch_police_to_pilot",dict(n=1,range=(200,700),alt=(2,25),speed=(0,8),path="wander",closest=(180,650),rf=True,rid="none",listed=False,target="none",launch="adjacent_lot",phase=["pregame"])),
(20,"C","Older drone with control link but no Remote ID","unauthorized_careless","dispatch_police_to_pilot",dict(n=1,range=(400,3000),alt=(20,120),speed=CONSUMER,path="wander",closest=(250,2500),rf=True,rid="none",listed=False,target="none")),
(21,"C","Unlicensed photographer shooting the crowd","unauthorized_careless","dispatch_police_to_pilot",dict(n=1,range=(200,800),alt=(30,100),speed=(0,8),path="orbit",closest=(120,500),rf=True,rid="valid",listed=False,target="none")),
(22,"C","Tourist flying 2 km out, unaware of restriction","unauthorized_careless","monitor",dict(n=1,range=(1500,5000),alt=(20,120),speed=CONSUMER,path="wander",closest=(1400,4800),rf=True,rid="valid",listed=False,target="none",launch="far")),
(23,"C","Fast racing-style drone flown for fun nearby","unauthorized_careless","dispatch_police_to_pilot",dict(n=1,range=(400,2000),alt=(2,60),speed=(22,40),path="erratic",closest=(300,1800),rf=True,rid="none",listed=False,target="none")),
(24,"D","Lost-link drone flying home across the stadium","borderline","alert_operator",dict(n=1,range=(600,3000),alt=(30,120),speed=(8,15),path="straight_rth",closest=(0,200),rf=False,rid=["valid","none"],listed=False,target="none")),
(25,"C","Protest drone towing a banner","unauthorized_careless","alert_operator",dict(n=1,range=(300,1500),alt=(30,100),speed=(3,10),path="direct_then_hover",closest=(0,150),rf=True,rid=["valid","none"],listed=False,target="bowl",payload=True)),
(26,"D","Authorized delivery drone drifting off route toward stadium","borderline","alert_operator",dict(n=1,range=(4000,7000),alt=(60,120),speed=DELIV,path="transit_then_deviate",closest=(300,2500),rf=True,rid="valid",listed=True,dev=(300,4000),target="none")),
(27,"D","Delivery drone with brief position glitch","borderline","monitor",dict(n=1,range=(5600,8000),alt=(60,120),speed=DELIV,path="transit",closest=(5600,7500),rf=True,rid="mismatch_brief",listed=True,dev=(0,30),target="none")),
(28,"D","Approved flight at the wrong time","borderline","alert_operator",dict(n=1,range=(800,5000),alt=(30,120),speed=(5,25),path="transit",closest=(500,4500),rf=True,rid="valid",listed="wrong_time",dev=(0,40),target="none")),
(29,"D","Authorized drone above its approved altitude","borderline","monitor",dict(n=1,range=(500,6000),alt=(130,300),speed=(5,25),path="transit",closest=(300,5500),rf=True,rid="valid",listed=True,dev=(0,40),target="none")),
(30,"D","Drone loitering at edge of restricted zone, repeatedly","borderline","alert_operator",dict(n=1,range=(5200,6000),alt=(40,120),speed=(0,12),path="edge_loiter",closest=(5000,5700),rf=[True,False],rid=["valid","none"],listed=False,target="none")),
(31,"D","Drone hovering at distance watching security positions","borderline","alert_operator",dict(n=1,range=(400,1500),alt=(60,120),speed=(0,3),path="hover_long",closest=(400,1500),rf=[True,False],rid="none",listed=False,target="none")),
(32,"D","Drone approaches then turns back (probing)","borderline","alert_operator",dict(n=1,range=(1500,5000),alt=(30,120),speed=CONSUMER,path="approach_retreat",closest=(250,1200),rf=[True,False],rid=["valid","none"],listed=False,target="none")),
(33,"D","Small mismatch between reported and radar position","borderline","monitor",dict(n=1,range=(600,6000),alt=(30,120),speed=(5,25),path="transit",closest=(400,5500),rf=True,rid="mismatch_small",listed=[True,False],dev=(0,40),target="none")),
(34,"D","Radio-silent drone passing by, not toward stadium","borderline","alert_operator",dict(n=1,range=(1500,5500),alt=(30,150),speed=(8,25),path="transit",closest=(900,4500),rf=False,rid="none",listed=False,target="none")),
(35,"D","Two unrelated hobbyists flying close together","borderline","dispatch_police_to_pilot",dict(n=2,range=(300,2000),alt=(20,120),speed=CONSUMER,path="wander_pair",closest=(200,1500),rf=True,rid=["valid","none"],listed=False,target="none")),
(36,"C","Hobbyist drone whose camera gimbal could be mistaken for a payload","unauthorized_careless","dispatch_police_to_pilot",dict(n=1,range=(300,2000),alt=(30,120),speed=(0,12),path="orbit",closest=(150,1200),rf=True,rid=["valid","none"],listed=False,target="none",payload=True)),
(37,"D","Unlit drone at night","borderline","alert_operator",dict(n=1,range=(400,4000),alt=(20,120),speed=CONSUMER,path="wander",closest=(250,3000),rf=[True,False],rid="none",listed=False,target="none",phase=["second_half","postgame"])),
(38,"D","Authorized drone with failing battery descending over stands","borderline","priority_alert",dict(n=1,range=(0,120),alt=(20,70),speed=(0,6),path="uncontrolled_descent",closest=(0,100),rf=True,rid="valid",listed=True,dev=(20,150),target="bowl",launch="inside_authorized")),
(39,"E","Radio-silent drone on fast direct approach","threat","priority_alert",dict(n=1,range=(1500,5500),alt=(30,120),speed=(15,30),path="direct",closest=(0,60),rf=False,rid="none",listed=False,target="bowl")),
(40,"E","Low approach hugging buildings and trees","threat","priority_alert",dict(n=1,range=(800,3500),alt=(5,25),speed=(8,22),path="direct_low",closest=(0,80),rf=[True,False],rid="none",listed=False,target="bowl")),
(41,"E","High approach then steep dive","threat","priority_alert",dict(n=1,range=(1500,5000),alt=(200,400),speed=(12,28),path="high_then_dive",closest=(0,60),rf=[True,False],rid="none",listed=False,target="bowl")),
(42,"E","Payload-carrying drone hovering over stands","threat","priority_alert",dict(n=1,range=(600,3000),alt=(30,90),speed=(6,18),path="direct_then_hover",closest=(0,90),rf=[True,False],rid="none",listed=False,target="bowl",payload=True)),
(43,"E","Payload dropped over the crowd","threat","priority_alert",dict(n=1,range=(600,3000),alt=(40,100),speed=(8,20),path="direct_drop_egress",closest=(0,80),rf=[True,False],rid="none",listed=False,target="bowl",payload=True)),
(44,"E","Launch from adjacent lot, over the rim in seconds","threat","priority_alert",dict(n=1,range=(150,450),alt=(0,5),speed=(10,22),path="pop_up_direct",closest=(0,60),rf=[True,False],rid="none",listed=False,target="bowl",launch="adjacent_lot")),
(45,"E","Launch from inside the stadium","threat","priority_alert",dict(n=1,range=(40,120),alt=(0,10),speed=(3,15),path="pop_up_direct",closest=(0,60),rf=True,rid="none",listed=False,target="bowl",launch="inside")),
(46,"E","Cloned Remote ID of an authorized drone","threat","priority_alert",dict(n=1,range=(1000,5000),alt=(30,120),speed=(8,25),path="direct",closest=(0,150),rf=True,rid="cloned",listed="cloned_id",target="bowl")),
(47,"E","Hijacked broadcast drone leaving its box","threat","priority_alert",dict(n=1,range=(0,80),alt=(30,70),speed=(4,18),path="box_then_break_out",closest=(0,100),rf=True,rid="valid",listed=True,dev=(60,300),target="bowl",launch="inside_authorized")),
(48,"E","Drone with falsified position","threat","priority_alert",dict(n=1,range=(1000,5000),alt=(30,120),speed=(8,25),path="direct",closest=(0,150),rf=True,rid="mismatch_large",listed=[True,False],target="bowl")),
(49,"E","Decoy on one side, real drone from the other","threat","priority_alert",dict(n=(2,4),range=(800,4000),alt=(20,120),speed=(8,25),path="decoy_plus_direct",closest=(0,80),rf=[True,False],rid="none",listed=False,target="bowl",swarm=True)),
(50,"E","Spotter loitering while a second drone attacks","threat","priority_alert",dict(n=2,range=(500,3500),alt=(30,120),speed=(5,25),path="spotter_plus_direct",closest=(0,80),rf=[True,False],rid="none",listed=False,target="bowl",swarm=True)),
(51,"E","Swarm converging from several directions","threat","priority_alert",dict(n=(3,30),range=(1200,5000),alt=(20,120),speed=(8,25),path="converge_multi",closest=(0,80),rf=[True,False],rid="none",listed=False,target="bowl",swarm=True)),
(52,"E","Swarm in formation that splits near the stadium","threat","priority_alert",dict(n=(3,30),range=(1500,5000),alt=(30,120),speed=(8,22),path="formation_then_split",closest=(0,100),rf=[True,False],rid="none",listed=False,target="bowl",swarm=True)),
(53,"E","Drones arriving in staggered waves","threat","priority_alert",dict(n=(4,24),range=(1500,5000),alt=(20,120),speed=(8,25),path="waves",closest=(0,100),rf=[True,False],rid="none",listed=False,target="bowl",swarm=True)),
(54,"E","Attack timed to the flyover","threat","priority_alert",dict(n=(1,3),range=(300,3000),alt=(10,120),speed=(10,28),path="direct",closest=(0,80),rf=[True,False],rid="none",listed=False,target="bowl",phase=["pregame"],concurrent="flyover")),
(55,"E","Attack on the tailgate area before the game","threat","priority_alert",dict(n=(1,4),range=(600,4000),alt=(10,100),speed=(8,25),path="direct",closest=(150,450),rf=[True,False],rid="none",listed=False,target="tailgate_lot",phase=["pregame"])),
(56,"E","Attack on the crowd leaving after the game","threat","priority_alert",dict(n=(1,4),range=(600,4000),alt=(10,100),speed=(8,25),path="direct",closest=(100,350),rf=[True,False],rid="none",listed=False,target="exit_plaza",phase=["postgame"])),
(57,"E","Fast fixed-wing drone","threat","priority_alert",dict(n=1,range=(2500,6000),alt=(40,200),speed=(25,45),path="direct",closest=(0,100),rf=[True,False],rid="none",listed=False,target="bowl",obj="fixed_wing_drone")),
(58,"E","Drone releasing powder or leaflets","threat","priority_alert",dict(n=1,range=(600,3000),alt=(30,90),speed=(5,15),path="direct_release_pass",closest=(0,90),rf=[True,False],rid="none",listed=False,target="bowl",payload=True)),
(68,"G","Light show drones arriving in staggered waves to staging (approved)","authorized","log_only",dict(n=(100,500),range=(300,1500),alt=(20,100),speed=(4,12),path="waves",closest=(0,100),rf=True,rid="valid",listed=True,dev=(0,15),target="none",phase=["halftime"],swarm=True)),
(69,"G","Approved drone making a fast direct pass over the bowl (planned shot)","authorized","log_only",dict(n=1,range=(300,1500),alt=(30,90),speed=(12,25),path="direct",closest=(0,60),rf=True,rid="valid",listed=True,dev=(0,20),target="none")),
(70,"G","Police drone diving fast toward an incident (authorized)","authorized","log_only",dict(n=1,range=(200,1200),alt=(60,150),speed=(15,28),path="high_then_dive",closest=(0,300),rf=True,rid="valid",listed=True,dev=(0,60),target="none")),
(71,"G","Approved ceremonial delivery flying straight into the bowl","authorized","log_only",dict(n=1,range=(400,2000),alt=(20,80),speed=(6,16),path="direct_then_hover",closest=(0,40),rf=True,rid="valid",listed=True,dev=(0,15),target="none",payload=True,phase=["pregame","halftime"])),
(72,"G","Approved camera drone hovering over the stands","authorized","log_only",dict(n=1,range=(0,150),alt=(30,80),speed=(0,4),path="hover_long",closest=(0,120),rf=True,rid="valid",listed=True,dev=(0,15),target="none",launch="inside_authorized")),
(73,"G","Approved broadcaster formation of several drones","authorized","log_only",dict(n=(3,6),range=(0,300),alt=(30,100),speed=(2,12),path="formation_then_split",closest=(0,150),rf=True,rid="valid",listed=True,dev=(0,20),target="none",swarm=True)),
(74,"G","Delivery-style corridor flight that is not on the approved list","borderline","alert_operator",dict(n=1,range=(4000,7000),alt=(60,120),speed=(14,28),path="transit",closest=(2500,5500),rf=True,rid=["valid","none"],listed=False,target="none")),
(75,"G","Light-show-style formation that is not scheduled","threat","priority_alert",dict(n=(20,200),range=(300,2500),alt=(30,120),speed=(2,10),path="formation_show",closest=(0,150),rf=[True,False],rid="none",listed=False,target="bowl",swarm=True)),
(76,"G","Extra unlisted drone hiding inside the approved light show","threat","priority_alert",dict(n=(1,3),range=(300,2500),alt=(40,120),speed=(5,15),path="join_formation",closest=(0,80),rf=[True,False],rid=["none","cloned"],listed=False,target="bowl",phase=["halftime"],concurrent="approved_light_show")),
(77,"G","Approved light show drone leaving formation toward the stands","borderline","priority_alert",dict(n=(1,5),range=(0,150),alt=(40,120),speed=(3,15),path="box_then_break_out",closest=(0,100),rf=True,rid="valid",listed=True,dev=(60,400),target="bowl",phase=["halftime"],concurrent="approved_light_show")),
(79,"D","Drone approaches, loiters near the edge, then leaves","borderline","alert_operator",dict(n=1,range=(800,3500),alt=(30,120),speed=(8,22),path="approach_loiter_retreat",closest=(150,500),rf=[True,False],rid="none",listed=False,target="none")),
(80,"E","Hijacked security drone leaving its patrol box","threat","priority_alert",dict(n=1,range=(0,80),alt=(30,90),speed=(8,25),path="box_then_break_out",closest=(0,100),rf=True,rid="valid",listed=True,dev=(60,300),target="bowl",launch="inside_authorized")),
(78,"G","Patrol-style orbit by a drone not on the approved list","borderline","alert_operator",dict(n=1,range=(150,900),alt=(40,110),speed=(3,15),path="orbit",closest=(120,600),rf=[True,False],rid=["valid","none"],listed=False,target="none")),
]
STRESS=[(59,"One radar offline",["radar_CCG","radar_WCG"]),(60,"RF sensor jammed or flooded",["rf_NE","rf_NW","rf_SE","rf_SW","rf_CCG","rf_WCG","rf_all"]),(61,"Camera blinded by sun or lights",["camera_NE","camera_NW","camera_SE","camera_SW","camera_CCG","camera_WCG"]),
(62,"Heavy rain degrading cameras",["cameras_all_rain"]),(63,"Sensors disagree on position",["position_conflict"]),(64,"Pod lost network link",["link_NE","link_NW","link_SE","link_SW","link_CCG","link_WCG"]),
(65,"Ten or more hobbyists at once",["operator_overload"]),(66,"Two drones cross, tracks may swap",["track_crossing"]),(67,"Drone hidden behind stands from most sensors",["occluded_by_stands"]),
(84,"Fog",["fog"]),(85,"Heavy RF interference",["rf_interference"]),(86,"Two cameras down",["cameras_out_2"]),(87,"Several objects arriving together",["crowded"]),
(88,"A whole sensor type switched off",["no_radar","no_rf","no_cam"])]
def pick(v,isint=False):
    if isinstance(v,tuple): return R.randint(v[0],v[1]) if isint else round(R.uniform(v[0],v[1]),1)
    if isinstance(v,list): return R.choice(v)
    return v
FIELDS=["case_id","scenario_id","group","scenario_name","expected_label","expected_response","is_swarm","object_type","n_objects","phase","lighting","weather","wind_mps","start_bearing_deg","start_range_m","start_alt_m","speed_mps","path_type","closest_approach_m","target","launch_site","rf_link","remote_id","on_approved_list","route_deviation_m","payload_visible","manifest_match","concurrent_event","degraded_sensor","stress_scenario","seed"]
def make(sc,stress=None):
    sid,grp,name,label,resp,p=sc
    phase=pick(p.get("phase",ANY)); n=pick(p.get("n",1),True); rng0=pick(p["range"]); close=min(pick(p["closest"]),rng0)
    weather=R.choices(["clear","wind","rain","fog"],[0.62,0.18,0.14,0.06])[0]
    row=dict(scenario_id=sid,group=grp,scenario_name=name,expected_label=label,expected_response=resp,is_swarm=bool(p.get("swarm",False)),object_type=p.get("obj","multirotor_drone"),n_objects=n,phase=phase,lighting=R.choice(LIGHT[phase]),weather=weather,
        wind_mps=round(R.uniform(6,14) if weather=="wind" else R.uniform(0,6),1),start_bearing_deg=round(R.uniform(0,360),1),start_range_m=rng0,start_alt_m=pick(p["alt"]),speed_mps=pick(p["speed"]),path_type=p["path"],closest_approach_m=close,
        target=p.get("target","none"),launch_site=pick(p.get("launch","far")),rf_link=pick(p.get("rf",False)),remote_id=pick(p.get("rid","none")),on_approved_list=pick(p.get("listed",False)),route_deviation_m=pick(p.get("dev",0)),
        payload_visible=bool(p.get("payload",False)),concurrent_event=p.get("concurrent","none"),degraded_sensor="none",stress_scenario="none",seed=R.randint(1,10**9))
    L=row["on_approved_list"]; d=row["route_deviation_m"]
    if label=="non_drone": mm="not_applicable"
    elif L is True: mm="matches_plan" if (d<=60 and row["start_alt_m"]<=125 or sid in (7,8,70)) else "deviates_from_plan"
    elif L=="wrong_time": mm="deviates_from_plan"
    elif L=="cloned_id" or row["remote_id"]=="cloned": mm="impersonates_approved"
    else: mm="not_on_manifest"
    if row["remote_id"] in ("mismatch_large",): mm="impersonates_approved" if L is True else mm
    row["manifest_match"]=mm
    if stress:
        row["degraded_sensor"]=R.choice(stress[2]); row["stress_scenario"]="%d: %s"%(stress[0],stress[1]); row["group"]="F"
        if stress[0]==62: row["weather"]="rain"
        if stress[0]==65:
            row["concurrent_event"]="%d_other_hobbyists"%R.randint(10,25)
        if stress[0]==66: row["concurrent_event"]="crossing_track"
        if stress[0]==87: row["concurrent_event"]="several_arrivals"
    return row
rows=[]; seen=set()
def add(r):
    k=tuple(r[f] for f in FIELDS[1:-1])
    if k in seen: return False
    seen.add(k); rows.append(r); return True
PER=45
# extra cases where the first model was weakest (v3)
EXTRA={24:250,32:250,79:200,41:200,47:300,80:200,76:250,77:200,26:150,18:250,38:150,46:150,
       42:250,43:150,58:250,36:150,21:100}   # round 2 (v4): payload threats vs camera-carrying hobbyists
for sc in S:
    c=0
    while c<PER+EXTRA.get(sc[0],0):
        if add(make(sc)): c+=1
base=[s for s in S if s[1] in ("C","D","E","G")]+[s for s in S if s[0] in (1,3,4)]
for st in STRESS:
    c=0
    while c<60:
        if add(make(R.choice(base),st)): c+=1
R.shuffle(rows)
for i,r in enumerate(rows): r["case_id"]="TC-%04d"%(i+1)
with open("cuas_test_cases.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
with open("cuas_scenarios.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["scenario_id","group","scenario_name","expected_label","expected_response","cases"])
    import collections; cnt=collections.Counter(r["scenario_id"] for r in rows if r["stress_scenario"]=="none")
    for s in S: w.writerow([s[0],s[1],s[2],s[3],s[4],cnt[s[0]]])
    sc2=collections.Counter(r["stress_scenario"] for r in rows if r["stress_scenario"]!="none")
    for st in STRESS: w.writerow([st[0],"F",st[1],"same as underlying scenario","same as underlying scenario",sc2["%d: %s"%(st[0],st[1])]])
import collections
print(len(rows)); print(collections.Counter(r["expected_label"] for r in rows)); print(collections.Counter(r["group"] for r in rows)); print(collections.Counter(r["expected_response"] for r in rows))
bad=[r for r in rows if r["closest_approach_m"]>r["start_range_m"] or r["speed_mps"]<0]; print("bad",len(bad))
