�
�(�(
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( ""
Ttype:
2	"
Tidxtype0:
2	
$
DisableCopyOnRead
resource�
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
�
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype�
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype�
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
�
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758š
�G
ConstConst*
_output_shapes	
:�*
dtype0	*�G
value�FB�F	�"�F                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      
�E
Const_1Const*
_output_shapes	
:�*
dtype0*�E
value�DB�D�B
yourselvesByours,ByourByoungerByou?Byou.Byou,Byou'reByou'dByou!ByouByears.ByearsBwriteBwouldn'tBwouldBworthyBworseBworlds.BworldsBworld.BworldBworks?Bworks.Bworked.BworkBwon'tBwithoutBwithBwishesBwindingBwillBwhyBwhoBwhimsBwhile,BwhileBwhichBwhereBwhenBwhateverBwhat?Bwhat'sBwhatBwere.BwereBwell,BwellBwelcome.B
weaponizedBweak.Bwe,Bwe'reBweBway.BwayBwavesB	watching,BwashBwasBwars,Bwar.Bwar,BwantedBwant.BwantBwannaBwait,BwaitBvoidBvirus.BviolentBvillainBversionsBversionB	variants.BvariantBvalueBusual.Busers,Bus?Bus...Bus.Bus,BusBup.Bup,BupBuntilB	universesBuniverseBuninitiated.Bundiscovered.Bunderstand.B
understandB
unchecked,Buh,BuhBtwoBturnsBturnedBturnBtryouts.BtryingBtryBtruth.Btrust.BtrustBtrueB	truckloadBtriedBtravelBtrade.BtouchesBtopBtootBtookBtoo.B	tomorrow,Btold.BtoldBto.BtoBtired,B	timeline,Btime.Btime,BtimeBtillBthrowsBthroughBthousandBthoughtBthoseBthis.BthisBthinkBthings.BthingsBthing.Bthing,BthingBthey'reBthey'llBtheyBtheseBthere,BthereBthen?BthenB
themselvesBthem?Bthem.Bthem,BthemBtheirs.BtheirBtheBthat?Bthat.B	that--notBthat,Bthat'sBthatBthanB	terrorismBtenBtellingBtellB
technologyBtechnicallyBtearsBteam?BteamBtakesBtakenBtake,BtakeBtableBsweetheart.BsurviveBsurroundingBsureB
superhumanBsuggestBstudy,B	strength.BstraightBstore.BstoppedBstopBstoodBstoneBstillBsticksBstayBstartsBstandBstackedBsqualor,BspreadBspotBspentBspecies.BspaceBsouth.Bsoul,Bsoon,BsonsBsongBson.BsonB	somethingBsomeBsolveB	solution.BsoleBsoldiersBsociety.BsocietyBso,BsoBsmileBsmellsBsmall,BsmallBslideBskies.BsittingBsitBsinsBsinceBsimpleBshowingBshowBshot,Bshoes.'BsharedBshareBsham.BshamBshakeBshadowsBsettle,B
serious?'.B	serious?'BseriousBselfishBself-defenseBself-congratulatoryBseenBseeBsecretBseaB
scroungingBscraps.Bscore.B	scientistBschoolB	schemers.BschemersBschemer.Bschemer,B	scenario.Bscars?Bsays,Bsay--BsayBsaved,BsavedBsaveBsandBsameB
salvation.BsafetyBsafe.BrunningBrunBruler,BruinsBruin.BroomBrockets.BrobBroads,Bright?BrightBrigBriddanceBrichBreverseB
revelationBrevealedBretiredB	resourcesBresourceB	reserves.B	research,BremindBrememberB
regardlessBreelection.BrecruitsBreckonBreceivedBreallyBrealizedBreality,B	realitiesBrealBreactionBraceBquit?Bquestioned,BputsBputBpurposeBpureBpublicBprotectBproof.B
principle,Bprice.BpriceBpreventBpressBpreserveB	preparingB	pregnant.BpreciousB	powerful,BpowerfulBpower?BpowerB	potentialBpossiblyBpositionBpoliticiansBplottingBplans.Bplans,Bplanet.BplanetBplan?Bplan".BplanBplague,Bplace,BpickB	personal,BpersonalBperiodBpeople.BpeopleBpeace.BpeaceBpayBpattern.BpatheticBpartsB	paradise.Bpanics,BpanicsBpal,Bown.BownBover.Bover,BoverBoutBourBothers.BotherBorganization,BorganismBorders?Border,BorderBorBopportunityB	operationBopenBonlyBone.Bone,BoneBonceBonBoldBokay?BoffBofBobsessedBobeyed.BobeyedBnowhere.Bnow...Bnow,Bnoticed?BnothingBnot.Bnot-BnotB
nostalgia.Bnose.'BnodBnobodyBnobleBno.BnoBnight,BniceBnext?BnewBneverBneedsBneedBnaturalBnamesBnamedBname?Bname.BmyselfBmyB	multiply,Bmove.Bmove,BmoveBmouth.BmostBmoreBmonopolyBmoney.B	mistaken.Bminds!BmincerBmillionBmightBmiddleBmethBmessedBmerelyBmenBmeetBmeantBme...Bme.Bme,Bme!BmeBmayorBmayBmatterBmarchBmany.BmanyBmanageBman.'Bman.Bman,BmanBmammals.BmammalBmamaBmakeBmaintained,BmaintainBmadeBmachineBloyalty?Bloved,BloveBlotBlosesBlookBlongerBlong,B	lobbying,BlivedBlive.BliveBlittleBlike,BlikeBlightB
lifetimes.Blife.BlifeBlevelBletBlessonBleftBleast,BlearningBlearnedBlearnBlaw.Blaw,BlaughingBlastBlandsBladiesBknowsBknownB
knowledge.Bknow?Bknow.Bknow,BknowBknife.BknifeBkitchenBkilledBkillBkid.BkeepBjustice.BjustBjudgedBjoiningBjoinBjerk.Bitself.BitsBit?Bit...Bit.Bit,Bit'sBitBissue.BisolatedBisn'tBisland,Bis?Bis.BisBintoB
instincts,BinstinctivelyB	innocent.B
initiated,B	ingenuityB	incoming.BinBimprove.BimproveB	importantB
importanceBifBidiotsBhungry?Bhungry.Bhungry,BhumansBhumanBhuh?BhowBhouseBhorrifying!Bhorn,BhopedBhome.BhomeBhmmm?BhistoryBhisBhim.BhimBhighlyBhighBherself.Bheroes.BheroesBheroBhere.B
here--whenBhere,BhereBher,BheldBheart.BheadsBheadBhe'sBheBhavingBhaveBhateBhasB	harnessedBharmony.BhardBhappensBhappenedBhair.'Bhadn'tBhadBguys!BguyBguns,BguessB
guaranteedBgroups--vigilantesBgroupBgrewBgreatestBgreaterBgotBgood,BgoodBgonnaBgoneBgoing.BgoingBgoesBgoBglobalBgivesBgiveB
girlfriendBgirlBgetsBgetB
gentlemen,B	gentlemenBgasBgangBgambit.BfurtherBfunny,BfullBfulfillBfuckingBfucked.BfuckedBfrom,BfromBfriends,BfriendBfreaksB	fragilityB
foundationBfoundBforBfoolBfollowsB	followingBflowBfirst.BfirstBfinite.Bfinite,Bfine.Bfine,BfineBfindBfillBfiguringBfightingBfightBfiend.BfewBfellas.Bfellas,BfeelingsBfear.BfatherBfateBfastBfarBfamily.Bfamily'sBfamilyBfameBfair!Bface.'Bface.BfaceB
extinctionBexplainBexperimentingB
expansion.Bexist.BexistBevolve.Bevil,Beverywhere.Beverything,B
everythingB	everyone.BeveryoneBeveryBeventsBevenBetc.BestablishedBeruptedBequilibriumB	epiphany.Benvironment,Benemy.BendedBend,BendBencounteredB	electric.B	efficientBeatBeasy,BeasyBears,BearnedBeachBduringBdubbedBdrumsBdrugsBdrugBdrinkerBdozenBdownsideBdon'tBdollars,Bdogma.BdogBdoesn'tBdoesBdoBdizzyBdivergeBdisease,B
discoveredBdisapprove?BdisappearedB	dinosaursBdinner.Bdigress.Bdie,Bdidn'tBdidB	dictator,BdiceBdevil,BdevelopsBdestroyB
destiny...Bdesires.BdeposeBdefendBdefeatedBdefeatBdecides.BdecidedB	deceptionBdead.BdeadBday.BdayB	daughtersBdarknessBdark.Bcure.Bcruel,BcrownB	crossfireBcreatureBcreatedBcrazierBcoupleBcountryBcould'veBcouldBcostBcosmicBcorrection.BcopsBconvenienceBcontrolBcontact.B	consumingB	consumed.B
conqueror.B
conquered.Bcommission.BcomesBcomeB	collapse.BcogBclearBclassifyBclarify.BcityBchoseBchildrenBchecked,BchasingBcharges.Bchaos?Bchaos.BchangeBceremonyBcentury.BceaseBcaughtBcataclysmicBcase.BcartelBcars.BcarryBcareBcapableBcannotBcancerBcanBcameBcallB	calculus.Bcage.BbyBbut,BbutBbusinessBburiedBbureaucracy.Bbullets.BbuiltBbuildBbuddy,BbuddyBbucketBbroughtBbrinkBbringBbreakB	branches.BbrainwashingBbrain'sBboy.BbothBborn.BbornBblownBblood.Bblinding...BbladeBbit.BbillionsBbiggestBbetweenBbetter?Bbetter.BbetterBbetrayedBbetrayBbest.BbestBbelongBbelliesBbelieveBbeingsBbeingBbeganBbeforeBbeenBbedBbecomesBbecomeBbecauseBbecameBbeast'sBbe.BbeBbattleBbangerBback.Bbaby.BawesomeBaway.B	authorityBattemptsBatBasshole?BasBarmory.Baren'tBarea.Barea,Bare.BareBanythingBanyBanother,BanotherB
annihilateBand,BandBanarchy.BanBamuseBam?Bam,BamBalwaysBalreadyBalongBalmostBally.Ball-outBallBalive,BalBahead,Bago,B
aggressiveBagesBagents...toBagent.BagentBageBagainstBafterBadoptedBaddedBactuallyBactivities.BactionBactBachieveB	accepted.BaboutBabilityB	abducted,BaB___BYourBYou'reBYouBWorkingB
Wonderful!B
WinterfellBWhoBWhichBWhenBWhat'sB	Westeros.BWell,BWeBWayne.BWardenBWar.BVictoryBUsingBUpsetBUltron:BUgh.BToBTime-KeepersBThoseBThisBThey'llBTheyBThere'sBThereBThen...IBThen,BTheatricalityBTheBThat'sBThatBTVA.BTVA,BStiflingB	StatesmanBStarks?BStarkBSo?BSo,BSoBSlideBShallBShadows.BShadows,B
Seriously,BSchemersBSansaBRooseBRightBRemains,BRealBRachelBRa'sBQuinjet.BPregnantBPeopleBPeaceBOneBOnceBOh.BOh,BNow,BNowBNothingBNotBNorthBNope.BNope,BNobodyBNoah.BNo.BNoB
Naturally,BNarcissistic,BMyBMultiversalBMr.BMoney'sBMoneyBMommyBMobBMerlin.BMakeBMIT,BLoveBLookBLittleBLifelongBLife.BLifeBLet'sBLetBLeagueBLannisters,B
Lannister.BKrypton.BKryptonBIt'sBItBIsn'tBIsB	IntroduceBInstead,BInBIf,BIfBI-I-IBI'veBI'mBI'llBI'dBIBHumanBHoweverBHowBHmm?BHi.BHey.BHence,BHenceBHell,BHellBHe'dBHeBHarvey.BHarryBHappyBGuysBGreat.BGrease.B
GratefullyB	Graffiti.BGordon'sBGood.BGoodBGoldenBGoingBGodBGoBGhul'sBGalahad.BFormedBForBFlash?BFlash.BFlash,BFlash!BFine.BExplainB
Excellent.BEveryBEvenBEonsBEasyBEarthB
DuplicatedBDon'tBDoesBDoBDays.BClimateBCircle.BCircle,BCharles,BCIABButBBruce?BBoltonBBingo!BBecauseBAtBAskBAngel,BAndBAmericanBAllBAliothBAh,BAgentBAfterB
Admirable.BAB50sB31stB20B...B'emB'causeB'WhyB'Thanks,B'Oh,B'Let'sB'IB"part
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
H
Const_3Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R 
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/Output_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/v/Output_Layer/bias
�
,Adam/v/Output_Layer/bias/Read/ReadVariableOpReadVariableOpAdam/v/Output_Layer/bias*
_output_shapes
:	*
dtype0
�
Adam/m/Output_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/m/Output_Layer/bias
�
,Adam/m/Output_Layer/bias/Read/ReadVariableOpReadVariableOpAdam/m/Output_Layer/bias*
_output_shapes
:	*
dtype0
�
Adam/v/Output_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
	*+
shared_nameAdam/v/Output_Layer/kernel
�
.Adam/v/Output_Layer/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Output_Layer/kernel*
_output_shapes

:
	*
dtype0
�
Adam/m/Output_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
	*+
shared_nameAdam/m/Output_Layer/kernel
�
.Adam/m/Output_Layer/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Output_Layer/kernel*
_output_shapes

:
	*
dtype0
�
Adam/v/Hidden_Layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/v/Hidden_Layer_2/bias
�
.Adam/v/Hidden_Layer_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/Hidden_Layer_2/bias*
_output_shapes
:
*
dtype0
�
Adam/m/Hidden_Layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameAdam/m/Hidden_Layer_2/bias
�
.Adam/m/Hidden_Layer_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/Hidden_Layer_2/bias*
_output_shapes
:
*
dtype0
�
Adam/v/Hidden_Layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*-
shared_nameAdam/v/Hidden_Layer_2/kernel
�
0Adam/v/Hidden_Layer_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Hidden_Layer_2/kernel*
_output_shapes

:
*
dtype0
�
Adam/m/Hidden_Layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*-
shared_nameAdam/m/Hidden_Layer_2/kernel
�
0Adam/m/Hidden_Layer_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Hidden_Layer_2/kernel*
_output_shapes

:
*
dtype0
�
Adam/v/Hidden_Layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/v/Hidden_Layer_1/bias
�
.Adam/v/Hidden_Layer_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/Hidden_Layer_1/bias*
_output_shapes
:*
dtype0
�
Adam/m/Hidden_Layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m/Hidden_Layer_1/bias
�
.Adam/m/Hidden_Layer_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/Hidden_Layer_1/bias*
_output_shapes
:*
dtype0
�
Adam/v/Hidden_Layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/v/Hidden_Layer_1/kernel
�
0Adam/v/Hidden_Layer_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Hidden_Layer_1/kernel*
_output_shapes

:*
dtype0
�
Adam/m/Hidden_Layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/m/Hidden_Layer_1/kernel
�
0Adam/m/Hidden_Layer_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Hidden_Layer_1/kernel*
_output_shapes

:*
dtype0
�
!Adam/v/Embedding_Layer/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*2
shared_name#!Adam/v/Embedding_Layer/embeddings
�
5Adam/v/Embedding_Layer/embeddings/Read/ReadVariableOpReadVariableOp!Adam/v/Embedding_Layer/embeddings*
_output_shapes
:	�*
dtype0
�
!Adam/m/Embedding_Layer/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*2
shared_name#!Adam/m/Embedding_Layer/embeddings
�
5Adam/m/Embedding_Layer/embeddings/Read/ReadVariableOpReadVariableOp!Adam/m/Embedding_Layer/embeddings*
_output_shapes
:	�*
dtype0
}
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_16*
value_dtype0	
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name222*
value_dtype0	
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
z
Output_Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameOutput_Layer/bias
s
%Output_Layer/bias/Read/ReadVariableOpReadVariableOpOutput_Layer/bias*
_output_shapes
:	*
dtype0
�
Output_Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
	*$
shared_nameOutput_Layer/kernel
{
'Output_Layer/kernel/Read/ReadVariableOpReadVariableOpOutput_Layer/kernel*
_output_shapes

:
	*
dtype0
~
Hidden_Layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameHidden_Layer_2/bias
w
'Hidden_Layer_2/bias/Read/ReadVariableOpReadVariableOpHidden_Layer_2/bias*
_output_shapes
:
*
dtype0
�
Hidden_Layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*&
shared_nameHidden_Layer_2/kernel

)Hidden_Layer_2/kernel/Read/ReadVariableOpReadVariableOpHidden_Layer_2/kernel*
_output_shapes

:
*
dtype0
~
Hidden_Layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameHidden_Layer_1/bias
w
'Hidden_Layer_1/bias/Read/ReadVariableOpReadVariableOpHidden_Layer_1/bias*
_output_shapes
:*
dtype0
�
Hidden_Layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameHidden_Layer_1/kernel

)Hidden_Layer_1/kernel/Read/ReadVariableOpReadVariableOpHidden_Layer_1/kernel*
_output_shapes

:*
dtype0
�
Embedding_Layer/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameEmbedding_Layer/embeddings
�
.Embedding_Layer/embeddings/Read/ReadVariableOpReadVariableOpEmbedding_Layer/embeddings*
_output_shapes
:	�*
dtype0
�
%serving_default_Text_Vectorizer_inputPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall%serving_default_Text_Vectorizer_input
hash_tableConst_4Const_3Const_5Embedding_Layer/embeddingsHidden_Layer_1/kernelHidden_Layer_1/biasHidden_Layer_2/kernelHidden_Layer_2/biasOutput_Layer/kernelOutput_Layer/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_3055
�
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__initializer_3580
�
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__initializer_3595
:
NoOpNoOp^PartitionedCall^StatefulPartitionedCall_1
�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
�<
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*�<
value�<B�< B�<
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	vectorize
	embedding
hidden_layer1
hidden_layer2
output_layer
	optimizer

signatures*
;
	keras_api
_lookup_layer
_adapt_function*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias*
5
1
2
 3
'4
(5
/6
07*
5
0
1
 2
'3
(4
/5
06*
* 
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
6trace_0
7trace_1
8trace_2
9trace_3* 
6
:trace_0
;trace_1
<trace_2
=trace_3* 
/
>	capture_1
?	capture_2
@	capture_3* 
�
A
_variables
B_iterations
C_learning_rate
D_index_dict
E
_momentums
F_velocities
G_update_step_xla*

Hserving_default* 
* 
7
I	keras_api
Jlookup_table
Ktoken_counts*

Ltrace_0* 

0*

0*
* 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Rtrace_0* 

Strace_0* 
nh
VARIABLE_VALUEEmbedding_Layer/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ytrace_0* 

Ztrace_0* 
e_
VARIABLE_VALUEHidden_Layer_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden_Layer_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

`trace_0* 

atrace_0* 
e_
VARIABLE_VALUEHidden_Layer_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden_Layer_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
c]
VARIABLE_VALUEOutput_Layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEOutput_Layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

i0
j1*
* 
* 
/
>	capture_1
?	capture_2
@	capture_3* 
/
>	capture_1
?	capture_2
@	capture_3* 
/
>	capture_1
?	capture_2
@	capture_3* 
/
>	capture_1
?	capture_2
@	capture_3* 
/
>	capture_1
?	capture_2
@	capture_3* 
/
>	capture_1
?	capture_2
@	capture_3* 
/
>	capture_1
?	capture_2
@	capture_3* 
/
>	capture_1
?	capture_2
@	capture_3* 
* 
* 
* 
r
B0
k1
l2
m3
n4
o5
p6
q7
r8
s9
t10
u11
v12
w13
x14*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
k0
m1
o2
q3
s4
u5
w6*
5
l0
n1
p2
r3
t4
v5
x6*
* 
/
>	capture_1
?	capture_2
@	capture_3* 
* 
R
y_initializer
z_create_resource
{_initialize
|_destroy_resource* 
�
}_create_resource
~_initialize
_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*

�	capture_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
lf
VARIABLE_VALUE!Adam/m/Embedding_Layer/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/v/Embedding_Layer/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/Hidden_Layer_1/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/Hidden_Layer_1/kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/Hidden_Layer_1/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/Hidden_Layer_1/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/Hidden_Layer_2/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/Hidden_Layer_2/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/Hidden_Layer_2/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/Hidden_Layer_2/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/Output_Layer/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/Output_Layer/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/Output_Layer/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/Output_Layer/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
"
�	capture_1
�	capture_2* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameEmbedding_Layer/embeddingsHidden_Layer_1/kernelHidden_Layer_1/biasHidden_Layer_2/kernelHidden_Layer_2/biasOutput_Layer/kernelOutput_Layer/bias	iterationlearning_rate!Adam/m/Embedding_Layer/embeddings!Adam/v/Embedding_Layer/embeddingsAdam/m/Hidden_Layer_1/kernelAdam/v/Hidden_Layer_1/kernelAdam/m/Hidden_Layer_1/biasAdam/v/Hidden_Layer_1/biasAdam/m/Hidden_Layer_2/kernelAdam/v/Hidden_Layer_2/kernelAdam/m/Hidden_Layer_2/biasAdam/v/Hidden_Layer_2/biasAdam/m/Output_Layer/kernelAdam/v/Output_Layer/kernelAdam/m/Output_Layer/biasAdam/v/Output_Layer/biastotal_1count_1totalcount?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1Const_6**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_3826
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameEmbedding_Layer/embeddingsHidden_Layer_1/kernelHidden_Layer_1/biasHidden_Layer_2/kernelHidden_Layer_2/biasOutput_Layer/kernelOutput_Layer/bias	iterationlearning_rateMutableHashTable!Adam/m/Embedding_Layer/embeddings!Adam/v/Embedding_Layer/embeddingsAdam/m/Hidden_Layer_1/kernelAdam/v/Hidden_Layer_1/kernelAdam/m/Hidden_Layer_1/biasAdam/v/Hidden_Layer_1/biasAdam/m/Hidden_Layer_2/kernelAdam/v/Hidden_Layer_2/kernelAdam/m/Hidden_Layer_2/biasAdam/v/Hidden_Layer_2/biasAdam/m/Output_Layer/kernelAdam/v/Output_Layer/kernelAdam/m/Output_Layer/biasAdam/v/Output_Layer/biastotal_1count_1totalcount*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_3920��
��
�
__inference__wrapped_model_2492
text_vectorizer_inpute
aembedding_analysis_model_text_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handlef
bembedding_analysis_model_text_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value	B
>embedding_analysis_model_text_vectorizer_string_lookup_equal_yE
Aembedding_analysis_model_text_vectorizer_string_lookup_selectv2_t	Q
>embedding_analysis_model_embedding_layer_embedding_lookup_2405:	�[
Iembedding_analysis_model_hidden_layer_1_tensordot_readvariableop_resource:U
Gembedding_analysis_model_hidden_layer_1_biasadd_readvariableop_resource:[
Iembedding_analysis_model_hidden_layer_2_tensordot_readvariableop_resource:
U
Gembedding_analysis_model_hidden_layer_2_biasadd_readvariableop_resource:
Y
Gembedding_analysis_model_output_layer_tensordot_readvariableop_resource:
	S
Eembedding_analysis_model_output_layer_biasadd_readvariableop_resource:	
identity��9Embedding_Analysis_Model/Embedding_Layer/embedding_lookup�>Embedding_Analysis_Model/Hidden_Layer_1/BiasAdd/ReadVariableOp�@Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/ReadVariableOp�>Embedding_Analysis_Model/Hidden_Layer_2/BiasAdd/ReadVariableOp�@Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/ReadVariableOp�<Embedding_Analysis_Model/Output_Layer/BiasAdd/ReadVariableOp�>Embedding_Analysis_Model/Output_Layer/Tensordot/ReadVariableOp�TEmbedding_Analysis_Model/Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2{
:Embedding_Analysis_Model/Text_Vectorizer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
BEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/StringSplitV2StringSplitV2text_vectorizer_inputCEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
HEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
JEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
JEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
BEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_sliceStridedSliceLEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/StringSplitV2:indices:0QEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice/stack:output:0SEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice/stack_1:output:0SEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
JEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
LEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
LEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
DEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice_1StridedSliceJEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/StringSplitV2:shape:0SEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice_1/stack:output:0UEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice_1/stack_1:output:0UEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
kEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastKEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
mEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastMEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
uEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeoEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
uEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
tEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProd~Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0~Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
yEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
wEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreater}Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0�Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
tEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast{Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
wEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
sEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxoEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0�Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
uEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2|Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0~Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
sEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulxEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0wEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
wEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumqEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0wEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
wEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumqEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0{Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
wEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
}Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
wEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeoEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0�Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
xEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount�Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0{Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0�Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
rEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
mEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0{Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
vEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
rEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
mEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0sEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0{Embedding_Analysis_Model/Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
TEmbedding_Analysis_Model/Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2aembedding_analysis_model_text_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handleKEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/StringSplitV2:values:0bembedding_analysis_model_text_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
<Embedding_Analysis_Model/Text_Vectorizer/string_lookup/EqualEqualKEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/StringSplitV2:values:0>embedding_analysis_model_text_vectorizer_string_lookup_equal_y*
T0*#
_output_shapes
:����������
?Embedding_Analysis_Model/Text_Vectorizer/string_lookup/SelectV2SelectV2@Embedding_Analysis_Model/Text_Vectorizer/string_lookup/Equal:z:0Aembedding_analysis_model_text_vectorizer_string_lookup_selectv2_t]Embedding_Analysis_Model/Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
?Embedding_Analysis_Model/Text_Vectorizer/string_lookup/IdentityIdentityHEmbedding_Analysis_Model/Text_Vectorizer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:����������
EEmbedding_Analysis_Model/Text_Vectorizer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
=Embedding_Analysis_Model/Text_Vectorizer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������	       �
LEmbedding_Analysis_Model/Text_Vectorizer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorFEmbedding_Analysis_Model/Text_Vectorizer/RaggedToTensor/Const:output:0HEmbedding_Analysis_Model/Text_Vectorizer/string_lookup/Identity:output:0NEmbedding_Analysis_Model/Text_Vectorizer/RaggedToTensor/default_value:output:0MEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice_1:output:0KEmbedding_Analysis_Model/Text_Vectorizer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������	*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
9Embedding_Analysis_Model/Embedding_Layer/embedding_lookupResourceGather>embedding_analysis_model_embedding_layer_embedding_lookup_2405UEmbedding_Analysis_Model/Text_Vectorizer/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*Q
_classG
ECloc:@Embedding_Analysis_Model/Embedding_Layer/embedding_lookup/2405*+
_output_shapes
:���������	*
dtype0�
BEmbedding_Analysis_Model/Embedding_Layer/embedding_lookup/IdentityIdentityBEmbedding_Analysis_Model/Embedding_Layer/embedding_lookup:output:0*
T0*Q
_classG
ECloc:@Embedding_Analysis_Model/Embedding_Layer/embedding_lookup/2405*+
_output_shapes
:���������	�
DEmbedding_Analysis_Model/Embedding_Layer/embedding_lookup/Identity_1IdentityKEmbedding_Analysis_Model/Embedding_Layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
@Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/ReadVariableOpReadVariableOpIembedding_analysis_model_hidden_layer_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
6Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
6Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
7Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/ShapeShapeMEmbedding_Analysis_Model/Embedding_Layer/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
::���
?Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/GatherV2GatherV2@Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/Shape:output:0?Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/free:output:0HEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
AEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/GatherV2_1GatherV2@Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/Shape:output:0?Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/axes:output:0JEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
7Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
6Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/ProdProdCEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/GatherV2:output:0@Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
9Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/Prod_1ProdEEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/GatherV2_1:output:0BEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
=Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/concatConcatV2?Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/free:output:0?Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/axes:output:0FEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
7Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/stackPack?Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/Prod:output:0AEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
;Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/transpose	TransposeMEmbedding_Analysis_Model/Embedding_Layer/embedding_lookup/Identity_1:output:0AEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
9Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/ReshapeReshape?Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/transpose:y:0@Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
8Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/MatMulMatMulBEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/Reshape:output:0HEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
9Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�
?Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/concat_1ConcatV2CEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/GatherV2:output:0BEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/Const_2:output:0HEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
1Embedding_Analysis_Model/Hidden_Layer_1/TensordotReshapeBEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/MatMul:product:0CEmbedding_Analysis_Model/Hidden_Layer_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
>Embedding_Analysis_Model/Hidden_Layer_1/BiasAdd/ReadVariableOpReadVariableOpGembedding_analysis_model_hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
/Embedding_Analysis_Model/Hidden_Layer_1/BiasAddBiasAdd:Embedding_Analysis_Model/Hidden_Layer_1/Tensordot:output:0FEmbedding_Analysis_Model/Hidden_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	�
,Embedding_Analysis_Model/Hidden_Layer_1/ReluRelu8Embedding_Analysis_Model/Hidden_Layer_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
@Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/ReadVariableOpReadVariableOpIembedding_analysis_model_hidden_layer_2_tensordot_readvariableop_resource*
_output_shapes

:
*
dtype0�
6Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
6Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
7Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/ShapeShape:Embedding_Analysis_Model/Hidden_Layer_1/Relu:activations:0*
T0*
_output_shapes
::���
?Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/GatherV2GatherV2@Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/Shape:output:0?Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/free:output:0HEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
AEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/GatherV2_1GatherV2@Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/Shape:output:0?Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/axes:output:0JEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
7Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
6Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/ProdProdCEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/GatherV2:output:0@Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
9Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
8Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/Prod_1ProdEEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/GatherV2_1:output:0BEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
=Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/concatConcatV2?Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/free:output:0?Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/axes:output:0FEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
7Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/stackPack?Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/Prod:output:0AEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
;Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/transpose	Transpose:Embedding_Analysis_Model/Hidden_Layer_1/Relu:activations:0AEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
9Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/ReshapeReshape?Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/transpose:y:0@Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
8Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/MatMulMatMulBEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/Reshape:output:0HEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
9Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
�
?Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/concat_1ConcatV2CEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/GatherV2:output:0BEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/Const_2:output:0HEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
1Embedding_Analysis_Model/Hidden_Layer_2/TensordotReshapeBEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/MatMul:product:0CEmbedding_Analysis_Model/Hidden_Layer_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	
�
>Embedding_Analysis_Model/Hidden_Layer_2/BiasAdd/ReadVariableOpReadVariableOpGembedding_analysis_model_hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
/Embedding_Analysis_Model/Hidden_Layer_2/BiasAddBiasAdd:Embedding_Analysis_Model/Hidden_Layer_2/Tensordot:output:0FEmbedding_Analysis_Model/Hidden_Layer_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	
�
/Embedding_Analysis_Model/Hidden_Layer_2/SigmoidSigmoid8Embedding_Analysis_Model/Hidden_Layer_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������	
�
>Embedding_Analysis_Model/Output_Layer/Tensordot/ReadVariableOpReadVariableOpGembedding_analysis_model_output_layer_tensordot_readvariableop_resource*
_output_shapes

:
	*
dtype0~
4Embedding_Analysis_Model/Output_Layer/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
4Embedding_Analysis_Model/Output_Layer/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
5Embedding_Analysis_Model/Output_Layer/Tensordot/ShapeShape3Embedding_Analysis_Model/Hidden_Layer_2/Sigmoid:y:0*
T0*
_output_shapes
::��
=Embedding_Analysis_Model/Output_Layer/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8Embedding_Analysis_Model/Output_Layer/Tensordot/GatherV2GatherV2>Embedding_Analysis_Model/Output_Layer/Tensordot/Shape:output:0=Embedding_Analysis_Model/Output_Layer/Tensordot/free:output:0FEmbedding_Analysis_Model/Output_Layer/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
?Embedding_Analysis_Model/Output_Layer/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:Embedding_Analysis_Model/Output_Layer/Tensordot/GatherV2_1GatherV2>Embedding_Analysis_Model/Output_Layer/Tensordot/Shape:output:0=Embedding_Analysis_Model/Output_Layer/Tensordot/axes:output:0HEmbedding_Analysis_Model/Output_Layer/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
5Embedding_Analysis_Model/Output_Layer/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
4Embedding_Analysis_Model/Output_Layer/Tensordot/ProdProdAEmbedding_Analysis_Model/Output_Layer/Tensordot/GatherV2:output:0>Embedding_Analysis_Model/Output_Layer/Tensordot/Const:output:0*
T0*
_output_shapes
: �
7Embedding_Analysis_Model/Output_Layer/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
6Embedding_Analysis_Model/Output_Layer/Tensordot/Prod_1ProdCEmbedding_Analysis_Model/Output_Layer/Tensordot/GatherV2_1:output:0@Embedding_Analysis_Model/Output_Layer/Tensordot/Const_1:output:0*
T0*
_output_shapes
: }
;Embedding_Analysis_Model/Output_Layer/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6Embedding_Analysis_Model/Output_Layer/Tensordot/concatConcatV2=Embedding_Analysis_Model/Output_Layer/Tensordot/free:output:0=Embedding_Analysis_Model/Output_Layer/Tensordot/axes:output:0DEmbedding_Analysis_Model/Output_Layer/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
5Embedding_Analysis_Model/Output_Layer/Tensordot/stackPack=Embedding_Analysis_Model/Output_Layer/Tensordot/Prod:output:0?Embedding_Analysis_Model/Output_Layer/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
9Embedding_Analysis_Model/Output_Layer/Tensordot/transpose	Transpose3Embedding_Analysis_Model/Hidden_Layer_2/Sigmoid:y:0?Embedding_Analysis_Model/Output_Layer/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	
�
7Embedding_Analysis_Model/Output_Layer/Tensordot/ReshapeReshape=Embedding_Analysis_Model/Output_Layer/Tensordot/transpose:y:0>Embedding_Analysis_Model/Output_Layer/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
6Embedding_Analysis_Model/Output_Layer/Tensordot/MatMulMatMul@Embedding_Analysis_Model/Output_Layer/Tensordot/Reshape:output:0FEmbedding_Analysis_Model/Output_Layer/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
7Embedding_Analysis_Model/Output_Layer/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	
=Embedding_Analysis_Model/Output_Layer/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8Embedding_Analysis_Model/Output_Layer/Tensordot/concat_1ConcatV2AEmbedding_Analysis_Model/Output_Layer/Tensordot/GatherV2:output:0@Embedding_Analysis_Model/Output_Layer/Tensordot/Const_2:output:0FEmbedding_Analysis_Model/Output_Layer/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
/Embedding_Analysis_Model/Output_Layer/TensordotReshape@Embedding_Analysis_Model/Output_Layer/Tensordot/MatMul:product:0AEmbedding_Analysis_Model/Output_Layer/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������		�
<Embedding_Analysis_Model/Output_Layer/BiasAdd/ReadVariableOpReadVariableOpEembedding_analysis_model_output_layer_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
-Embedding_Analysis_Model/Output_Layer/BiasAddBiasAdd8Embedding_Analysis_Model/Output_Layer/Tensordot:output:0DEmbedding_Analysis_Model/Output_Layer/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������		�
-Embedding_Analysis_Model/Output_Layer/SoftmaxSoftmax6Embedding_Analysis_Model/Output_Layer/BiasAdd:output:0*
T0*+
_output_shapes
:���������		�
IdentityIdentity7Embedding_Analysis_Model/Output_Layer/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:���������		�
NoOpNoOp:^Embedding_Analysis_Model/Embedding_Layer/embedding_lookup?^Embedding_Analysis_Model/Hidden_Layer_1/BiasAdd/ReadVariableOpA^Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/ReadVariableOp?^Embedding_Analysis_Model/Hidden_Layer_2/BiasAdd/ReadVariableOpA^Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/ReadVariableOp=^Embedding_Analysis_Model/Output_Layer/BiasAdd/ReadVariableOp?^Embedding_Analysis_Model/Output_Layer/Tensordot/ReadVariableOpU^Embedding_Analysis_Model/Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 2v
9Embedding_Analysis_Model/Embedding_Layer/embedding_lookup9Embedding_Analysis_Model/Embedding_Layer/embedding_lookup2�
>Embedding_Analysis_Model/Hidden_Layer_1/BiasAdd/ReadVariableOp>Embedding_Analysis_Model/Hidden_Layer_1/BiasAdd/ReadVariableOp2�
@Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/ReadVariableOp@Embedding_Analysis_Model/Hidden_Layer_1/Tensordot/ReadVariableOp2�
>Embedding_Analysis_Model/Hidden_Layer_2/BiasAdd/ReadVariableOp>Embedding_Analysis_Model/Hidden_Layer_2/BiasAdd/ReadVariableOp2�
@Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/ReadVariableOp@Embedding_Analysis_Model/Hidden_Layer_2/Tensordot/ReadVariableOp2|
<Embedding_Analysis_Model/Output_Layer/BiasAdd/ReadVariableOp<Embedding_Analysis_Model/Output_Layer/BiasAdd/ReadVariableOp2�
>Embedding_Analysis_Model/Output_Layer/Tensordot/ReadVariableOp>Embedding_Analysis_Model/Output_Layer/Tensordot/ReadVariableOp2�
TEmbedding_Analysis_Model/Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2TEmbedding_Analysis_Model/Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:���������
/
_user_specified_nameText_Vectorizer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�{
�
 __inference__traced_restore_3920
file_prefix>
+assignvariableop_embedding_layer_embeddings:	�:
(assignvariableop_1_hidden_layer_1_kernel:4
&assignvariableop_2_hidden_layer_1_bias::
(assignvariableop_3_hidden_layer_2_kernel:
4
&assignvariableop_4_hidden_layer_2_bias:
8
&assignvariableop_5_output_layer_kernel:
	2
$assignvariableop_6_output_layer_bias:	&
assignvariableop_7_iteration:	 *
 assignvariableop_8_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: G
4assignvariableop_9_adam_m_embedding_layer_embeddings:	�H
5assignvariableop_10_adam_v_embedding_layer_embeddings:	�B
0assignvariableop_11_adam_m_hidden_layer_1_kernel:B
0assignvariableop_12_adam_v_hidden_layer_1_kernel:<
.assignvariableop_13_adam_m_hidden_layer_1_bias:<
.assignvariableop_14_adam_v_hidden_layer_1_bias:B
0assignvariableop_15_adam_m_hidden_layer_2_kernel:
B
0assignvariableop_16_adam_v_hidden_layer_2_kernel:
<
.assignvariableop_17_adam_m_hidden_layer_2_bias:
<
.assignvariableop_18_adam_v_hidden_layer_2_bias:
@
.assignvariableop_19_adam_m_output_layer_kernel:
	@
.assignvariableop_20_adam_v_output_layer_kernel:
	:
,assignvariableop_21_adam_m_output_layer_bias:	:
,assignvariableop_22_adam_v_output_layer_bias:	%
assignvariableop_23_total_1: %
assignvariableop_24_count_1: #
assignvariableop_25_total: #
assignvariableop_26_count: 
identity_28��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�2MutableHashTable_table_restore/LookupTableImportV2�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp+assignvariableop_embedding_layer_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp(assignvariableop_1_hidden_layer_1_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_hidden_layer_1_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp(assignvariableop_3_hidden_layer_2_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_hidden_layer_2_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_output_layer_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp$assignvariableop_6_output_layer_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_iterationIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_learning_rateIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0�
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:9RestoreV2:tensors:10*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*&
 _has_manual_control_dependencies(*
_output_shapes
 ^

Identity_9IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp4assignvariableop_9_adam_m_embedding_layer_embeddingsIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_adam_v_embedding_layer_embeddingsIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp0assignvariableop_11_adam_m_hidden_layer_1_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp0assignvariableop_12_adam_v_hidden_layer_1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp.assignvariableop_13_adam_m_hidden_layer_1_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_adam_v_hidden_layer_1_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_adam_m_hidden_layer_2_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp0assignvariableop_16_adam_v_hidden_layer_2_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_adam_m_hidden_layer_2_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp.assignvariableop_18_adam_v_hidden_layer_2_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_m_output_layer_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_v_output_layer_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_m_output_layer_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_v_output_layer_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)
%
#
_class
loc:@MutableHashTable
�
�
7__inference_Embedding_Analysis_Model_layer_call_fn_2930
text_vectorizer_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:
	
	unknown_9:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_vectorizer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2905s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:���������
/
_user_specified_nameText_Vectorizer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_Hidden_Layer_1_layer_call_fn_3456

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_2587s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_2587

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������	e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������	z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
9
__inference__creator_3572
identity��
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name222*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�
�
__inference__initializer_35806
2key_value_init221_lookuptableimportv2_table_handle.
*key_value_init221_lookuptableimportv2_keys0
,key_value_init221_lookuptableimportv2_values	
identity��%key_value_init221/LookupTableImportV2�
%key_value_init221/LookupTableImportV2LookupTableImportV22key_value_init221_lookuptableimportv2_table_handle*key_value_init221_lookuptableimportv2_keys,key_value_init221_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init221/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :�:�2N
%key_value_init221/LookupTableImportV2%key_value_init221/LookupTableImportV2:!

_output_shapes	
:�:!

_output_shapes	
:�
�
�
F__inference_Output_Layer_layer_call_and_return_conditional_losses_3567

inputs3
!tensordot_readvariableop_resource:
	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:
	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	
�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������		r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������		Z
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:���������		d
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:���������		z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	

 
_user_specified_nameinputs
�
+
__inference__destroyer_3585
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�j
�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2737
text_vectorizer_inputL
Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handleM
Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value	)
%text_vectorizer_string_lookup_equal_y,
(text_vectorizer_string_lookup_selectv2_t	'
embedding_layer_2718:	�%
hidden_layer_1_2721:!
hidden_layer_1_2723:%
hidden_layer_2_2726:
!
hidden_layer_2_2728:
#
output_layer_2731:
	
output_layer_2733:	
identity��'Embedding_Layer/StatefulPartitionedCall�&Hidden_Layer_1/StatefulPartitionedCall�&Hidden_Layer_2/StatefulPartitionedCall�$Output_Layer/StatefulPartitionedCall�;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2b
!Text_Vectorizer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
)Text_Vectorizer/StringSplit/StringSplitV2StringSplitV2text_vectorizer_input*Text_Vectorizer/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
/Text_Vectorizer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
1Text_Vectorizer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
1Text_Vectorizer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
)Text_Vectorizer/StringSplit/strided_sliceStridedSlice3Text_Vectorizer/StringSplit/StringSplitV2:indices:08Text_Vectorizer/StringSplit/strided_slice/stack:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_1:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask{
1Text_Vectorizer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Text_Vectorizer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Text_Vectorizer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+Text_Vectorizer/StringSplit/strided_slice_1StridedSlice1Text_Vectorizer/StringSplit/StringSplitV2:shape:0:Text_Vectorizer/StringSplit/strided_slice_1/stack:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_1:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
RText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast2Text_Vectorizer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast4Text_Vectorizer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdeText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
`Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterdText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0iText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastbText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2cText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
dText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountgText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumfText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
]Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2fText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handle2Text_Vectorizer/StringSplit/StringSplitV2:values:0Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
#Text_Vectorizer/string_lookup/EqualEqual2Text_Vectorizer/StringSplit/StringSplitV2:values:0%text_vectorizer_string_lookup_equal_y*
T0*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/SelectV2SelectV2'Text_Vectorizer/string_lookup/Equal:z:0(text_vectorizer_string_lookup_selectv2_tDText_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/IdentityIdentity/Text_Vectorizer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������n
,Text_Vectorizer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R }
$Text_Vectorizer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������	       �
3Text_Vectorizer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-Text_Vectorizer/RaggedToTensor/Const:output:0/Text_Vectorizer/string_lookup/Identity:output:05Text_Vectorizer/RaggedToTensor/default_value:output:04Text_Vectorizer/StringSplit/strided_slice_1:output:02Text_Vectorizer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������	*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
'Embedding_Layer/StatefulPartitionedCallStatefulPartitionedCall<Text_Vectorizer/RaggedToTensor/RaggedTensorToTensor:result:0embedding_layer_2718*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_2552�
&Hidden_Layer_1/StatefulPartitionedCallStatefulPartitionedCall0Embedding_Layer/StatefulPartitionedCall:output:0hidden_layer_1_2721hidden_layer_1_2723*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_2587�
&Hidden_Layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_Layer_1/StatefulPartitionedCall:output:0hidden_layer_2_2726hidden_layer_2_2728*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_2624�
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_Layer_2/StatefulPartitionedCall:output:0output_layer_2731output_layer_2733*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_Output_Layer_layer_call_and_return_conditional_losses_2661�
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		�
NoOpNoOp(^Embedding_Layer/StatefulPartitionedCall'^Hidden_Layer_1/StatefulPartitionedCall'^Hidden_Layer_2/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall<^Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 2R
'Embedding_Layer/StatefulPartitionedCall'Embedding_Layer/StatefulPartitionedCall2P
&Hidden_Layer_1/StatefulPartitionedCall&Hidden_Layer_1/StatefulPartitionedCall2P
&Hidden_Layer_2/StatefulPartitionedCall&Hidden_Layer_2/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall2z
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:���������
/
_user_specified_nameText_Vectorizer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
7__inference_Embedding_Analysis_Model_layer_call_fn_3130

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:
	
	unknown_9:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2809s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
"__inference_signature_wrapper_3055
text_vectorizer_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:
	
	unknown_9:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_vectorizer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_2492s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:���������
/
_user_specified_nameText_Vectorizer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
7__inference_Embedding_Analysis_Model_layer_call_fn_3157

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:
	
	unknown_9:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2905s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
+
__inference__destroyer_3600
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�j
�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2809

inputsL
Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handleM
Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value	)
%text_vectorizer_string_lookup_equal_y,
(text_vectorizer_string_lookup_selectv2_t	'
embedding_layer_2790:	�%
hidden_layer_1_2793:!
hidden_layer_1_2795:%
hidden_layer_2_2798:
!
hidden_layer_2_2800:
#
output_layer_2803:
	
output_layer_2805:	
identity��'Embedding_Layer/StatefulPartitionedCall�&Hidden_Layer_1/StatefulPartitionedCall�&Hidden_Layer_2/StatefulPartitionedCall�$Output_Layer/StatefulPartitionedCall�;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2b
!Text_Vectorizer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
)Text_Vectorizer/StringSplit/StringSplitV2StringSplitV2inputs*Text_Vectorizer/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
/Text_Vectorizer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
1Text_Vectorizer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
1Text_Vectorizer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
)Text_Vectorizer/StringSplit/strided_sliceStridedSlice3Text_Vectorizer/StringSplit/StringSplitV2:indices:08Text_Vectorizer/StringSplit/strided_slice/stack:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_1:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask{
1Text_Vectorizer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Text_Vectorizer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Text_Vectorizer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+Text_Vectorizer/StringSplit/strided_slice_1StridedSlice1Text_Vectorizer/StringSplit/StringSplitV2:shape:0:Text_Vectorizer/StringSplit/strided_slice_1/stack:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_1:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
RText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast2Text_Vectorizer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast4Text_Vectorizer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdeText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
`Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterdText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0iText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastbText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2cText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
dText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountgText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumfText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
]Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2fText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handle2Text_Vectorizer/StringSplit/StringSplitV2:values:0Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
#Text_Vectorizer/string_lookup/EqualEqual2Text_Vectorizer/StringSplit/StringSplitV2:values:0%text_vectorizer_string_lookup_equal_y*
T0*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/SelectV2SelectV2'Text_Vectorizer/string_lookup/Equal:z:0(text_vectorizer_string_lookup_selectv2_tDText_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/IdentityIdentity/Text_Vectorizer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������n
,Text_Vectorizer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R }
$Text_Vectorizer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������	       �
3Text_Vectorizer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-Text_Vectorizer/RaggedToTensor/Const:output:0/Text_Vectorizer/string_lookup/Identity:output:05Text_Vectorizer/RaggedToTensor/default_value:output:04Text_Vectorizer/StringSplit/strided_slice_1:output:02Text_Vectorizer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������	*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
'Embedding_Layer/StatefulPartitionedCallStatefulPartitionedCall<Text_Vectorizer/RaggedToTensor/RaggedTensorToTensor:result:0embedding_layer_2790*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_2552�
&Hidden_Layer_1/StatefulPartitionedCallStatefulPartitionedCall0Embedding_Layer/StatefulPartitionedCall:output:0hidden_layer_1_2793hidden_layer_1_2795*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_2587�
&Hidden_Layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_Layer_1/StatefulPartitionedCall:output:0hidden_layer_2_2798hidden_layer_2_2800*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_2624�
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_Layer_2/StatefulPartitionedCall:output:0output_layer_2803output_layer_2805*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_Output_Layer_layer_call_and_return_conditional_losses_2661�
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		�
NoOpNoOp(^Embedding_Layer/StatefulPartitionedCall'^Hidden_Layer_1/StatefulPartitionedCall'^Hidden_Layer_2/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall<^Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 2R
'Embedding_Layer/StatefulPartitionedCall'Embedding_Layer/StatefulPartitionedCall2P
&Hidden_Layer_1/StatefulPartitionedCall&Hidden_Layer_1/StatefulPartitionedCall2P
&Hidden_Layer_2/StatefulPartitionedCall&Hidden_Layer_2/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall2z
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�j
�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2905

inputsL
Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handleM
Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value	)
%text_vectorizer_string_lookup_equal_y,
(text_vectorizer_string_lookup_selectv2_t	'
embedding_layer_2886:	�%
hidden_layer_1_2889:!
hidden_layer_1_2891:%
hidden_layer_2_2894:
!
hidden_layer_2_2896:
#
output_layer_2899:
	
output_layer_2901:	
identity��'Embedding_Layer/StatefulPartitionedCall�&Hidden_Layer_1/StatefulPartitionedCall�&Hidden_Layer_2/StatefulPartitionedCall�$Output_Layer/StatefulPartitionedCall�;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2b
!Text_Vectorizer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
)Text_Vectorizer/StringSplit/StringSplitV2StringSplitV2inputs*Text_Vectorizer/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
/Text_Vectorizer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
1Text_Vectorizer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
1Text_Vectorizer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
)Text_Vectorizer/StringSplit/strided_sliceStridedSlice3Text_Vectorizer/StringSplit/StringSplitV2:indices:08Text_Vectorizer/StringSplit/strided_slice/stack:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_1:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask{
1Text_Vectorizer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Text_Vectorizer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Text_Vectorizer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+Text_Vectorizer/StringSplit/strided_slice_1StridedSlice1Text_Vectorizer/StringSplit/StringSplitV2:shape:0:Text_Vectorizer/StringSplit/strided_slice_1/stack:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_1:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
RText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast2Text_Vectorizer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast4Text_Vectorizer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdeText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
`Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterdText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0iText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastbText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2cText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
dText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountgText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumfText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
]Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2fText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handle2Text_Vectorizer/StringSplit/StringSplitV2:values:0Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
#Text_Vectorizer/string_lookup/EqualEqual2Text_Vectorizer/StringSplit/StringSplitV2:values:0%text_vectorizer_string_lookup_equal_y*
T0*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/SelectV2SelectV2'Text_Vectorizer/string_lookup/Equal:z:0(text_vectorizer_string_lookup_selectv2_tDText_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/IdentityIdentity/Text_Vectorizer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������n
,Text_Vectorizer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R }
$Text_Vectorizer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������	       �
3Text_Vectorizer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-Text_Vectorizer/RaggedToTensor/Const:output:0/Text_Vectorizer/string_lookup/Identity:output:05Text_Vectorizer/RaggedToTensor/default_value:output:04Text_Vectorizer/StringSplit/strided_slice_1:output:02Text_Vectorizer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������	*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
'Embedding_Layer/StatefulPartitionedCallStatefulPartitionedCall<Text_Vectorizer/RaggedToTensor/RaggedTensorToTensor:result:0embedding_layer_2886*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_2552�
&Hidden_Layer_1/StatefulPartitionedCallStatefulPartitionedCall0Embedding_Layer/StatefulPartitionedCall:output:0hidden_layer_1_2889hidden_layer_1_2891*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_2587�
&Hidden_Layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_Layer_1/StatefulPartitionedCall:output:0hidden_layer_2_2894hidden_layer_2_2896*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_2624�
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_Layer_2/StatefulPartitionedCall:output:0output_layer_2899output_layer_2901*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_Output_Layer_layer_call_and_return_conditional_losses_2661�
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		�
NoOpNoOp(^Embedding_Layer/StatefulPartitionedCall'^Hidden_Layer_1/StatefulPartitionedCall'^Hidden_Layer_2/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall<^Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 2R
'Embedding_Layer/StatefulPartitionedCall'Embedding_Layer/StatefulPartitionedCall2P
&Hidden_Layer_1/StatefulPartitionedCall&Hidden_Layer_1/StatefulPartitionedCall2P
&Hidden_Layer_2/StatefulPartitionedCall&Hidden_Layer_2/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall2z
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_Hidden_Layer_2_layer_call_fn_3496

inputs
unknown:

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_2624s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
+__inference_Output_Layer_layer_call_fn_3536

inputs
unknown:
	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_Output_Layer_layer_call_and_return_conditional_losses_2661s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������	

 
_user_specified_nameinputs
��
�
__inference__traced_save_3826
file_prefixD
1read_disablecopyonread_embedding_layer_embeddings:	�@
.read_1_disablecopyonread_hidden_layer_1_kernel::
,read_2_disablecopyonread_hidden_layer_1_bias:@
.read_3_disablecopyonread_hidden_layer_2_kernel:
:
,read_4_disablecopyonread_hidden_layer_2_bias:
>
,read_5_disablecopyonread_output_layer_kernel:
	8
*read_6_disablecopyonread_output_layer_bias:	,
"read_7_disablecopyonread_iteration:	 0
&read_8_disablecopyonread_learning_rate: M
:read_9_disablecopyonread_adam_m_embedding_layer_embeddings:	�N
;read_10_disablecopyonread_adam_v_embedding_layer_embeddings:	�H
6read_11_disablecopyonread_adam_m_hidden_layer_1_kernel:H
6read_12_disablecopyonread_adam_v_hidden_layer_1_kernel:B
4read_13_disablecopyonread_adam_m_hidden_layer_1_bias:B
4read_14_disablecopyonread_adam_v_hidden_layer_1_bias:H
6read_15_disablecopyonread_adam_m_hidden_layer_2_kernel:
H
6read_16_disablecopyonread_adam_v_hidden_layer_2_kernel:
B
4read_17_disablecopyonread_adam_m_hidden_layer_2_bias:
B
4read_18_disablecopyonread_adam_v_hidden_layer_2_bias:
F
4read_19_disablecopyonread_adam_m_output_layer_kernel:
	F
4read_20_disablecopyonread_adam_v_output_layer_kernel:
	@
2read_21_disablecopyonread_adam_m_output_layer_bias:	@
2read_22_disablecopyonread_adam_v_output_layer_bias:	+
!read_23_disablecopyonread_total_1: +
!read_24_disablecopyonread_count_1: )
read_25_disablecopyonread_total: )
read_26_disablecopyonread_count: J
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	
savev2_const_6
identity_55��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead1read_disablecopyonread_embedding_layer_embeddings"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp1read_disablecopyonread_embedding_layer_embeddings^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_1/DisableCopyOnReadDisableCopyOnRead.read_1_disablecopyonread_hidden_layer_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp.read_1_disablecopyonread_hidden_layer_1_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_2/DisableCopyOnReadDisableCopyOnRead,read_2_disablecopyonread_hidden_layer_1_bias"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp,read_2_disablecopyonread_hidden_layer_1_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_3/DisableCopyOnReadDisableCopyOnRead.read_3_disablecopyonread_hidden_layer_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp.read_3_disablecopyonread_hidden_layer_2_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_4/DisableCopyOnReadDisableCopyOnRead,read_4_disablecopyonread_hidden_layer_2_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp,read_4_disablecopyonread_hidden_layer_2_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_5/DisableCopyOnReadDisableCopyOnRead,read_5_disablecopyonread_output_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp,read_5_disablecopyonread_output_layer_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
	*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
	e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:
	~
Read_6/DisableCopyOnReadDisableCopyOnRead*read_6_disablecopyonread_output_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp*read_6_disablecopyonread_output_layer_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	v
Read_7/DisableCopyOnReadDisableCopyOnRead"read_7_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp"read_7_disablecopyonread_iteration^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_learning_rate^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead:read_9_disablecopyonread_adam_m_embedding_layer_embeddings"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp:read_9_disablecopyonread_adam_m_embedding_layer_embeddings^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_10/DisableCopyOnReadDisableCopyOnRead;read_10_disablecopyonread_adam_v_embedding_layer_embeddings"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp;read_10_disablecopyonread_adam_v_embedding_layer_embeddings^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_11/DisableCopyOnReadDisableCopyOnRead6read_11_disablecopyonread_adam_m_hidden_layer_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp6read_11_disablecopyonread_adam_m_hidden_layer_1_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_12/DisableCopyOnReadDisableCopyOnRead6read_12_disablecopyonread_adam_v_hidden_layer_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp6read_12_disablecopyonread_adam_v_hidden_layer_1_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_13/DisableCopyOnReadDisableCopyOnRead4read_13_disablecopyonread_adam_m_hidden_layer_1_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp4read_13_disablecopyonread_adam_m_hidden_layer_1_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnRead4read_14_disablecopyonread_adam_v_hidden_layer_1_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp4read_14_disablecopyonread_adam_v_hidden_layer_1_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead6read_15_disablecopyonread_adam_m_hidden_layer_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp6read_15_disablecopyonread_adam_m_hidden_layer_2_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_16/DisableCopyOnReadDisableCopyOnRead6read_16_disablecopyonread_adam_v_hidden_layer_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp6read_16_disablecopyonread_adam_v_hidden_layer_2_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:
�
Read_17/DisableCopyOnReadDisableCopyOnRead4read_17_disablecopyonread_adam_m_hidden_layer_2_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp4read_17_disablecopyonread_adam_m_hidden_layer_2_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_18/DisableCopyOnReadDisableCopyOnRead4read_18_disablecopyonread_adam_v_hidden_layer_2_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp4read_18_disablecopyonread_adam_v_hidden_layer_2_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:
�
Read_19/DisableCopyOnReadDisableCopyOnRead4read_19_disablecopyonread_adam_m_output_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp4read_19_disablecopyonread_adam_m_output_layer_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
	*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
	e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:
	�
Read_20/DisableCopyOnReadDisableCopyOnRead4read_20_disablecopyonread_adam_v_output_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp4read_20_disablecopyonread_adam_v_output_layer_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
	*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
	e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:
	�
Read_21/DisableCopyOnReadDisableCopyOnRead2read_21_disablecopyonread_adam_m_output_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp2read_21_disablecopyonread_adam_m_output_layer_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
Read_22/DisableCopyOnReadDisableCopyOnRead2read_22_disablecopyonread_adam_v_output_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp2read_22_disablecopyonread_adam_v_output_layer_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	v
Read_23/DisableCopyOnReadDisableCopyOnRead!read_23_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp!read_23_disablecopyonread_total_1^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_24/DisableCopyOnReadDisableCopyOnRead!read_24_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp!read_24_disablecopyonread_count_1^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_25/DisableCopyOnReadDisableCopyOnReadread_25_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOpread_25_disablecopyonread_total^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_26/DisableCopyOnReadDisableCopyOnReadread_26_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOpread_26_disablecopyonread_count^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0savev2_const_6"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *,
dtypes"
 2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_54Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_55IdentityIdentity_54:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_55Identity_55:output:0*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : ::: 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
::

_output_shapes
::

_output_shapes
: 
�
-
__inference__initializer_3595
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
__inference_save_fn_3619
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	��?MutableHashTable_lookup_table_export_values/LookupTableExportV2�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: �

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: �

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:�
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
��
�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_3294

inputsL
Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handleM
Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value	)
%text_vectorizer_string_lookup_equal_y,
(text_vectorizer_string_lookup_selectv2_t	8
%embedding_layer_embedding_lookup_3207:	�B
0hidden_layer_1_tensordot_readvariableop_resource:<
.hidden_layer_1_biasadd_readvariableop_resource:B
0hidden_layer_2_tensordot_readvariableop_resource:
<
.hidden_layer_2_biasadd_readvariableop_resource:
@
.output_layer_tensordot_readvariableop_resource:
	:
,output_layer_biasadd_readvariableop_resource:	
identity�� Embedding_Layer/embedding_lookup�%Hidden_Layer_1/BiasAdd/ReadVariableOp�'Hidden_Layer_1/Tensordot/ReadVariableOp�%Hidden_Layer_2/BiasAdd/ReadVariableOp�'Hidden_Layer_2/Tensordot/ReadVariableOp�#Output_Layer/BiasAdd/ReadVariableOp�%Output_Layer/Tensordot/ReadVariableOp�;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2b
!Text_Vectorizer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
)Text_Vectorizer/StringSplit/StringSplitV2StringSplitV2inputs*Text_Vectorizer/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
/Text_Vectorizer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
1Text_Vectorizer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
1Text_Vectorizer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
)Text_Vectorizer/StringSplit/strided_sliceStridedSlice3Text_Vectorizer/StringSplit/StringSplitV2:indices:08Text_Vectorizer/StringSplit/strided_slice/stack:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_1:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask{
1Text_Vectorizer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Text_Vectorizer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Text_Vectorizer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+Text_Vectorizer/StringSplit/strided_slice_1StridedSlice1Text_Vectorizer/StringSplit/StringSplitV2:shape:0:Text_Vectorizer/StringSplit/strided_slice_1/stack:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_1:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
RText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast2Text_Vectorizer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast4Text_Vectorizer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdeText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
`Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterdText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0iText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastbText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2cText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
dText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountgText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumfText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
]Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2fText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handle2Text_Vectorizer/StringSplit/StringSplitV2:values:0Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
#Text_Vectorizer/string_lookup/EqualEqual2Text_Vectorizer/StringSplit/StringSplitV2:values:0%text_vectorizer_string_lookup_equal_y*
T0*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/SelectV2SelectV2'Text_Vectorizer/string_lookup/Equal:z:0(text_vectorizer_string_lookup_selectv2_tDText_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/IdentityIdentity/Text_Vectorizer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������n
,Text_Vectorizer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R }
$Text_Vectorizer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������	       �
3Text_Vectorizer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-Text_Vectorizer/RaggedToTensor/Const:output:0/Text_Vectorizer/string_lookup/Identity:output:05Text_Vectorizer/RaggedToTensor/default_value:output:04Text_Vectorizer/StringSplit/strided_slice_1:output:02Text_Vectorizer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������	*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
 Embedding_Layer/embedding_lookupResourceGather%embedding_layer_embedding_lookup_3207<Text_Vectorizer/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*8
_class.
,*loc:@Embedding_Layer/embedding_lookup/3207*+
_output_shapes
:���������	*
dtype0�
)Embedding_Layer/embedding_lookup/IdentityIdentity)Embedding_Layer/embedding_lookup:output:0*
T0*8
_class.
,*loc:@Embedding_Layer/embedding_lookup/3207*+
_output_shapes
:���������	�
+Embedding_Layer/embedding_lookup/Identity_1Identity2Embedding_Layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
'Hidden_Layer_1/Tensordot/ReadVariableOpReadVariableOp0hidden_layer_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0g
Hidden_Layer_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
Hidden_Layer_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Hidden_Layer_1/Tensordot/ShapeShape4Embedding_Layer/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
::��h
&Hidden_Layer_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Hidden_Layer_1/Tensordot/GatherV2GatherV2'Hidden_Layer_1/Tensordot/Shape:output:0&Hidden_Layer_1/Tensordot/free:output:0/Hidden_Layer_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(Hidden_Layer_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#Hidden_Layer_1/Tensordot/GatherV2_1GatherV2'Hidden_Layer_1/Tensordot/Shape:output:0&Hidden_Layer_1/Tensordot/axes:output:01Hidden_Layer_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
Hidden_Layer_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Hidden_Layer_1/Tensordot/ProdProd*Hidden_Layer_1/Tensordot/GatherV2:output:0'Hidden_Layer_1/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 Hidden_Layer_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Hidden_Layer_1/Tensordot/Prod_1Prod,Hidden_Layer_1/Tensordot/GatherV2_1:output:0)Hidden_Layer_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$Hidden_Layer_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Hidden_Layer_1/Tensordot/concatConcatV2&Hidden_Layer_1/Tensordot/free:output:0&Hidden_Layer_1/Tensordot/axes:output:0-Hidden_Layer_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Hidden_Layer_1/Tensordot/stackPack&Hidden_Layer_1/Tensordot/Prod:output:0(Hidden_Layer_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
"Hidden_Layer_1/Tensordot/transpose	Transpose4Embedding_Layer/embedding_lookup/Identity_1:output:0(Hidden_Layer_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
 Hidden_Layer_1/Tensordot/ReshapeReshape&Hidden_Layer_1/Tensordot/transpose:y:0'Hidden_Layer_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Hidden_Layer_1/Tensordot/MatMulMatMul)Hidden_Layer_1/Tensordot/Reshape:output:0/Hidden_Layer_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
 Hidden_Layer_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&Hidden_Layer_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Hidden_Layer_1/Tensordot/concat_1ConcatV2*Hidden_Layer_1/Tensordot/GatherV2:output:0)Hidden_Layer_1/Tensordot/Const_2:output:0/Hidden_Layer_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Hidden_Layer_1/TensordotReshape)Hidden_Layer_1/Tensordot/MatMul:product:0*Hidden_Layer_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
%Hidden_Layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Hidden_Layer_1/BiasAddBiasAdd!Hidden_Layer_1/Tensordot:output:0-Hidden_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	r
Hidden_Layer_1/ReluReluHidden_Layer_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
'Hidden_Layer_2/Tensordot/ReadVariableOpReadVariableOp0hidden_layer_2_tensordot_readvariableop_resource*
_output_shapes

:
*
dtype0g
Hidden_Layer_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
Hidden_Layer_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
Hidden_Layer_2/Tensordot/ShapeShape!Hidden_Layer_1/Relu:activations:0*
T0*
_output_shapes
::��h
&Hidden_Layer_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Hidden_Layer_2/Tensordot/GatherV2GatherV2'Hidden_Layer_2/Tensordot/Shape:output:0&Hidden_Layer_2/Tensordot/free:output:0/Hidden_Layer_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(Hidden_Layer_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#Hidden_Layer_2/Tensordot/GatherV2_1GatherV2'Hidden_Layer_2/Tensordot/Shape:output:0&Hidden_Layer_2/Tensordot/axes:output:01Hidden_Layer_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
Hidden_Layer_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Hidden_Layer_2/Tensordot/ProdProd*Hidden_Layer_2/Tensordot/GatherV2:output:0'Hidden_Layer_2/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 Hidden_Layer_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Hidden_Layer_2/Tensordot/Prod_1Prod,Hidden_Layer_2/Tensordot/GatherV2_1:output:0)Hidden_Layer_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$Hidden_Layer_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Hidden_Layer_2/Tensordot/concatConcatV2&Hidden_Layer_2/Tensordot/free:output:0&Hidden_Layer_2/Tensordot/axes:output:0-Hidden_Layer_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Hidden_Layer_2/Tensordot/stackPack&Hidden_Layer_2/Tensordot/Prod:output:0(Hidden_Layer_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
"Hidden_Layer_2/Tensordot/transpose	Transpose!Hidden_Layer_1/Relu:activations:0(Hidden_Layer_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
 Hidden_Layer_2/Tensordot/ReshapeReshape&Hidden_Layer_2/Tensordot/transpose:y:0'Hidden_Layer_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Hidden_Layer_2/Tensordot/MatMulMatMul)Hidden_Layer_2/Tensordot/Reshape:output:0/Hidden_Layer_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
 Hidden_Layer_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
h
&Hidden_Layer_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Hidden_Layer_2/Tensordot/concat_1ConcatV2*Hidden_Layer_2/Tensordot/GatherV2:output:0)Hidden_Layer_2/Tensordot/Const_2:output:0/Hidden_Layer_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Hidden_Layer_2/TensordotReshape)Hidden_Layer_2/Tensordot/MatMul:product:0*Hidden_Layer_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	
�
%Hidden_Layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
Hidden_Layer_2/BiasAddBiasAdd!Hidden_Layer_2/Tensordot:output:0-Hidden_Layer_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	
x
Hidden_Layer_2/SigmoidSigmoidHidden_Layer_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������	
�
%Output_Layer/Tensordot/ReadVariableOpReadVariableOp.output_layer_tensordot_readvariableop_resource*
_output_shapes

:
	*
dtype0e
Output_Layer/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
Output_Layer/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       t
Output_Layer/Tensordot/ShapeShapeHidden_Layer_2/Sigmoid:y:0*
T0*
_output_shapes
::��f
$Output_Layer/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_Layer/Tensordot/GatherV2GatherV2%Output_Layer/Tensordot/Shape:output:0$Output_Layer/Tensordot/free:output:0-Output_Layer/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
&Output_Layer/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Output_Layer/Tensordot/GatherV2_1GatherV2%Output_Layer/Tensordot/Shape:output:0$Output_Layer/Tensordot/axes:output:0/Output_Layer/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:f
Output_Layer/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Output_Layer/Tensordot/ProdProd(Output_Layer/Tensordot/GatherV2:output:0%Output_Layer/Tensordot/Const:output:0*
T0*
_output_shapes
: h
Output_Layer/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Output_Layer/Tensordot/Prod_1Prod*Output_Layer/Tensordot/GatherV2_1:output:0'Output_Layer/Tensordot/Const_1:output:0*
T0*
_output_shapes
: d
"Output_Layer/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_Layer/Tensordot/concatConcatV2$Output_Layer/Tensordot/free:output:0$Output_Layer/Tensordot/axes:output:0+Output_Layer/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Output_Layer/Tensordot/stackPack$Output_Layer/Tensordot/Prod:output:0&Output_Layer/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
 Output_Layer/Tensordot/transpose	TransposeHidden_Layer_2/Sigmoid:y:0&Output_Layer/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	
�
Output_Layer/Tensordot/ReshapeReshape$Output_Layer/Tensordot/transpose:y:0%Output_Layer/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Output_Layer/Tensordot/MatMulMatMul'Output_Layer/Tensordot/Reshape:output:0-Output_Layer/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	h
Output_Layer/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	f
$Output_Layer/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_Layer/Tensordot/concat_1ConcatV2(Output_Layer/Tensordot/GatherV2:output:0'Output_Layer/Tensordot/Const_2:output:0-Output_Layer/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Output_Layer/TensordotReshape'Output_Layer/Tensordot/MatMul:product:0(Output_Layer/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������		�
#Output_Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Output_Layer/BiasAddBiasAddOutput_Layer/Tensordot:output:0+Output_Layer/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������		t
Output_Layer/SoftmaxSoftmaxOutput_Layer/BiasAdd:output:0*
T0*+
_output_shapes
:���������		q
IdentityIdentityOutput_Layer/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:���������		�
NoOpNoOp!^Embedding_Layer/embedding_lookup&^Hidden_Layer_1/BiasAdd/ReadVariableOp(^Hidden_Layer_1/Tensordot/ReadVariableOp&^Hidden_Layer_2/BiasAdd/ReadVariableOp(^Hidden_Layer_2/Tensordot/ReadVariableOp$^Output_Layer/BiasAdd/ReadVariableOp&^Output_Layer/Tensordot/ReadVariableOp<^Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 2D
 Embedding_Layer/embedding_lookup Embedding_Layer/embedding_lookup2N
%Hidden_Layer_1/BiasAdd/ReadVariableOp%Hidden_Layer_1/BiasAdd/ReadVariableOp2R
'Hidden_Layer_1/Tensordot/ReadVariableOp'Hidden_Layer_1/Tensordot/ReadVariableOp2N
%Hidden_Layer_2/BiasAdd/ReadVariableOp%Hidden_Layer_2/BiasAdd/ReadVariableOp2R
'Hidden_Layer_2/Tensordot/ReadVariableOp'Hidden_Layer_2/Tensordot/ReadVariableOp2J
#Output_Layer/BiasAdd/ReadVariableOp#Output_Layer/BiasAdd/ReadVariableOp2N
%Output_Layer/Tensordot/ReadVariableOp%Output_Layer/Tensordot/ReadVariableOp2z
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
��
�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_3431

inputsL
Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handleM
Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value	)
%text_vectorizer_string_lookup_equal_y,
(text_vectorizer_string_lookup_selectv2_t	8
%embedding_layer_embedding_lookup_3344:	�B
0hidden_layer_1_tensordot_readvariableop_resource:<
.hidden_layer_1_biasadd_readvariableop_resource:B
0hidden_layer_2_tensordot_readvariableop_resource:
<
.hidden_layer_2_biasadd_readvariableop_resource:
@
.output_layer_tensordot_readvariableop_resource:
	:
,output_layer_biasadd_readvariableop_resource:	
identity�� Embedding_Layer/embedding_lookup�%Hidden_Layer_1/BiasAdd/ReadVariableOp�'Hidden_Layer_1/Tensordot/ReadVariableOp�%Hidden_Layer_2/BiasAdd/ReadVariableOp�'Hidden_Layer_2/Tensordot/ReadVariableOp�#Output_Layer/BiasAdd/ReadVariableOp�%Output_Layer/Tensordot/ReadVariableOp�;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2b
!Text_Vectorizer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
)Text_Vectorizer/StringSplit/StringSplitV2StringSplitV2inputs*Text_Vectorizer/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
/Text_Vectorizer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
1Text_Vectorizer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
1Text_Vectorizer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
)Text_Vectorizer/StringSplit/strided_sliceStridedSlice3Text_Vectorizer/StringSplit/StringSplitV2:indices:08Text_Vectorizer/StringSplit/strided_slice/stack:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_1:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask{
1Text_Vectorizer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Text_Vectorizer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Text_Vectorizer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+Text_Vectorizer/StringSplit/strided_slice_1StridedSlice1Text_Vectorizer/StringSplit/StringSplitV2:shape:0:Text_Vectorizer/StringSplit/strided_slice_1/stack:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_1:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
RText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast2Text_Vectorizer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast4Text_Vectorizer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdeText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
`Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterdText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0iText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastbText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2cText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
dText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountgText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumfText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
]Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2fText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handle2Text_Vectorizer/StringSplit/StringSplitV2:values:0Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
#Text_Vectorizer/string_lookup/EqualEqual2Text_Vectorizer/StringSplit/StringSplitV2:values:0%text_vectorizer_string_lookup_equal_y*
T0*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/SelectV2SelectV2'Text_Vectorizer/string_lookup/Equal:z:0(text_vectorizer_string_lookup_selectv2_tDText_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/IdentityIdentity/Text_Vectorizer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������n
,Text_Vectorizer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R }
$Text_Vectorizer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������	       �
3Text_Vectorizer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-Text_Vectorizer/RaggedToTensor/Const:output:0/Text_Vectorizer/string_lookup/Identity:output:05Text_Vectorizer/RaggedToTensor/default_value:output:04Text_Vectorizer/StringSplit/strided_slice_1:output:02Text_Vectorizer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������	*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
 Embedding_Layer/embedding_lookupResourceGather%embedding_layer_embedding_lookup_3344<Text_Vectorizer/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*8
_class.
,*loc:@Embedding_Layer/embedding_lookup/3344*+
_output_shapes
:���������	*
dtype0�
)Embedding_Layer/embedding_lookup/IdentityIdentity)Embedding_Layer/embedding_lookup:output:0*
T0*8
_class.
,*loc:@Embedding_Layer/embedding_lookup/3344*+
_output_shapes
:���������	�
+Embedding_Layer/embedding_lookup/Identity_1Identity2Embedding_Layer/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	�
'Hidden_Layer_1/Tensordot/ReadVariableOpReadVariableOp0hidden_layer_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0g
Hidden_Layer_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
Hidden_Layer_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
Hidden_Layer_1/Tensordot/ShapeShape4Embedding_Layer/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
::��h
&Hidden_Layer_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Hidden_Layer_1/Tensordot/GatherV2GatherV2'Hidden_Layer_1/Tensordot/Shape:output:0&Hidden_Layer_1/Tensordot/free:output:0/Hidden_Layer_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(Hidden_Layer_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#Hidden_Layer_1/Tensordot/GatherV2_1GatherV2'Hidden_Layer_1/Tensordot/Shape:output:0&Hidden_Layer_1/Tensordot/axes:output:01Hidden_Layer_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
Hidden_Layer_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Hidden_Layer_1/Tensordot/ProdProd*Hidden_Layer_1/Tensordot/GatherV2:output:0'Hidden_Layer_1/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 Hidden_Layer_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Hidden_Layer_1/Tensordot/Prod_1Prod,Hidden_Layer_1/Tensordot/GatherV2_1:output:0)Hidden_Layer_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$Hidden_Layer_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Hidden_Layer_1/Tensordot/concatConcatV2&Hidden_Layer_1/Tensordot/free:output:0&Hidden_Layer_1/Tensordot/axes:output:0-Hidden_Layer_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Hidden_Layer_1/Tensordot/stackPack&Hidden_Layer_1/Tensordot/Prod:output:0(Hidden_Layer_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
"Hidden_Layer_1/Tensordot/transpose	Transpose4Embedding_Layer/embedding_lookup/Identity_1:output:0(Hidden_Layer_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
 Hidden_Layer_1/Tensordot/ReshapeReshape&Hidden_Layer_1/Tensordot/transpose:y:0'Hidden_Layer_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Hidden_Layer_1/Tensordot/MatMulMatMul)Hidden_Layer_1/Tensordot/Reshape:output:0/Hidden_Layer_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������j
 Hidden_Layer_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&Hidden_Layer_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Hidden_Layer_1/Tensordot/concat_1ConcatV2*Hidden_Layer_1/Tensordot/GatherV2:output:0)Hidden_Layer_1/Tensordot/Const_2:output:0/Hidden_Layer_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Hidden_Layer_1/TensordotReshape)Hidden_Layer_1/Tensordot/MatMul:product:0*Hidden_Layer_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	�
%Hidden_Layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Hidden_Layer_1/BiasAddBiasAdd!Hidden_Layer_1/Tensordot:output:0-Hidden_Layer_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	r
Hidden_Layer_1/ReluReluHidden_Layer_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������	�
'Hidden_Layer_2/Tensordot/ReadVariableOpReadVariableOp0hidden_layer_2_tensordot_readvariableop_resource*
_output_shapes

:
*
dtype0g
Hidden_Layer_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
Hidden_Layer_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
Hidden_Layer_2/Tensordot/ShapeShape!Hidden_Layer_1/Relu:activations:0*
T0*
_output_shapes
::��h
&Hidden_Layer_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Hidden_Layer_2/Tensordot/GatherV2GatherV2'Hidden_Layer_2/Tensordot/Shape:output:0&Hidden_Layer_2/Tensordot/free:output:0/Hidden_Layer_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(Hidden_Layer_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#Hidden_Layer_2/Tensordot/GatherV2_1GatherV2'Hidden_Layer_2/Tensordot/Shape:output:0&Hidden_Layer_2/Tensordot/axes:output:01Hidden_Layer_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
Hidden_Layer_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Hidden_Layer_2/Tensordot/ProdProd*Hidden_Layer_2/Tensordot/GatherV2:output:0'Hidden_Layer_2/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 Hidden_Layer_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Hidden_Layer_2/Tensordot/Prod_1Prod,Hidden_Layer_2/Tensordot/GatherV2_1:output:0)Hidden_Layer_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$Hidden_Layer_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Hidden_Layer_2/Tensordot/concatConcatV2&Hidden_Layer_2/Tensordot/free:output:0&Hidden_Layer_2/Tensordot/axes:output:0-Hidden_Layer_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Hidden_Layer_2/Tensordot/stackPack&Hidden_Layer_2/Tensordot/Prod:output:0(Hidden_Layer_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
"Hidden_Layer_2/Tensordot/transpose	Transpose!Hidden_Layer_1/Relu:activations:0(Hidden_Layer_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
 Hidden_Layer_2/Tensordot/ReshapeReshape&Hidden_Layer_2/Tensordot/transpose:y:0'Hidden_Layer_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Hidden_Layer_2/Tensordot/MatMulMatMul)Hidden_Layer_2/Tensordot/Reshape:output:0/Hidden_Layer_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
j
 Hidden_Layer_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
h
&Hidden_Layer_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Hidden_Layer_2/Tensordot/concat_1ConcatV2*Hidden_Layer_2/Tensordot/GatherV2:output:0)Hidden_Layer_2/Tensordot/Const_2:output:0/Hidden_Layer_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Hidden_Layer_2/TensordotReshape)Hidden_Layer_2/Tensordot/MatMul:product:0*Hidden_Layer_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	
�
%Hidden_Layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
Hidden_Layer_2/BiasAddBiasAdd!Hidden_Layer_2/Tensordot:output:0-Hidden_Layer_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	
x
Hidden_Layer_2/SigmoidSigmoidHidden_Layer_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������	
�
%Output_Layer/Tensordot/ReadVariableOpReadVariableOp.output_layer_tensordot_readvariableop_resource*
_output_shapes

:
	*
dtype0e
Output_Layer/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
Output_Layer/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       t
Output_Layer/Tensordot/ShapeShapeHidden_Layer_2/Sigmoid:y:0*
T0*
_output_shapes
::��f
$Output_Layer/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_Layer/Tensordot/GatherV2GatherV2%Output_Layer/Tensordot/Shape:output:0$Output_Layer/Tensordot/free:output:0-Output_Layer/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
&Output_Layer/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Output_Layer/Tensordot/GatherV2_1GatherV2%Output_Layer/Tensordot/Shape:output:0$Output_Layer/Tensordot/axes:output:0/Output_Layer/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:f
Output_Layer/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Output_Layer/Tensordot/ProdProd(Output_Layer/Tensordot/GatherV2:output:0%Output_Layer/Tensordot/Const:output:0*
T0*
_output_shapes
: h
Output_Layer/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Output_Layer/Tensordot/Prod_1Prod*Output_Layer/Tensordot/GatherV2_1:output:0'Output_Layer/Tensordot/Const_1:output:0*
T0*
_output_shapes
: d
"Output_Layer/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_Layer/Tensordot/concatConcatV2$Output_Layer/Tensordot/free:output:0$Output_Layer/Tensordot/axes:output:0+Output_Layer/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Output_Layer/Tensordot/stackPack$Output_Layer/Tensordot/Prod:output:0&Output_Layer/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
 Output_Layer/Tensordot/transpose	TransposeHidden_Layer_2/Sigmoid:y:0&Output_Layer/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������	
�
Output_Layer/Tensordot/ReshapeReshape$Output_Layer/Tensordot/transpose:y:0%Output_Layer/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Output_Layer/Tensordot/MatMulMatMul'Output_Layer/Tensordot/Reshape:output:0-Output_Layer/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	h
Output_Layer/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	f
$Output_Layer/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_Layer/Tensordot/concat_1ConcatV2(Output_Layer/Tensordot/GatherV2:output:0'Output_Layer/Tensordot/Const_2:output:0-Output_Layer/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Output_Layer/TensordotReshape'Output_Layer/Tensordot/MatMul:product:0(Output_Layer/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������		�
#Output_Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
Output_Layer/BiasAddBiasAddOutput_Layer/Tensordot:output:0+Output_Layer/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������		t
Output_Layer/SoftmaxSoftmaxOutput_Layer/BiasAdd:output:0*
T0*+
_output_shapes
:���������		q
IdentityIdentityOutput_Layer/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:���������		�
NoOpNoOp!^Embedding_Layer/embedding_lookup&^Hidden_Layer_1/BiasAdd/ReadVariableOp(^Hidden_Layer_1/Tensordot/ReadVariableOp&^Hidden_Layer_2/BiasAdd/ReadVariableOp(^Hidden_Layer_2/Tensordot/ReadVariableOp$^Output_Layer/BiasAdd/ReadVariableOp&^Output_Layer/Tensordot/ReadVariableOp<^Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 2D
 Embedding_Layer/embedding_lookup Embedding_Layer/embedding_lookup2N
%Hidden_Layer_1/BiasAdd/ReadVariableOp%Hidden_Layer_1/BiasAdd/ReadVariableOp2R
'Hidden_Layer_1/Tensordot/ReadVariableOp'Hidden_Layer_1/Tensordot/ReadVariableOp2N
%Hidden_Layer_2/BiasAdd/ReadVariableOp%Hidden_Layer_2/BiasAdd/ReadVariableOp2R
'Hidden_Layer_2/Tensordot/ReadVariableOp'Hidden_Layer_2/Tensordot/ReadVariableOp2J
#Output_Layer/BiasAdd/ReadVariableOp#Output_Layer/BiasAdd/ReadVariableOp2N
%Output_Layer/Tensordot/ReadVariableOp%Output_Layer/Tensordot/ReadVariableOp2z
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
E
__inference__creator_3590
identity: ��MutableHashTable}
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_16*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
�D
�
__inference_adapt_step_3103
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	��IteratorGetNext�(None_lookup_table_find/LookupTableFindV2�,None_lookup_table_insert/LookupTableInsertV2�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
2R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
StringSplit/StringSplitV2StringSplitV2IteratorGetNext:components:0StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountWStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:���������:���������:���������*
out_idx0	�
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:�
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
�
�
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_3487

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������	e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������	z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
__inference_restore_fn_3627
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity��2MutableHashTable_table_restore/LookupTableImportV2�
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
�
�
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_3527

inputs3
!tensordot_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:
*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	
Z
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:���������	
^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:���������	
z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_2624

inputs3
!tensordot_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:
*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������	
Z
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:���������	
^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:���������	
z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_2552

inputs	(
embedding_lookup_2546:	�
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_2546inputs*
Tindices0	*(
_class
loc:@embedding_lookup/2546*+
_output_shapes
:���������	*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/2546*+
_output_shapes
:���������	�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������	Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������	: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�j
�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2668
text_vectorizer_inputL
Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handleM
Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value	)
%text_vectorizer_string_lookup_equal_y,
(text_vectorizer_string_lookup_selectv2_t	'
embedding_layer_2553:	�%
hidden_layer_1_2588:!
hidden_layer_1_2590:%
hidden_layer_2_2625:
!
hidden_layer_2_2627:
#
output_layer_2662:
	
output_layer_2664:	
identity��'Embedding_Layer/StatefulPartitionedCall�&Hidden_Layer_1/StatefulPartitionedCall�&Hidden_Layer_2/StatefulPartitionedCall�$Output_Layer/StatefulPartitionedCall�;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2b
!Text_Vectorizer/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
)Text_Vectorizer/StringSplit/StringSplitV2StringSplitV2text_vectorizer_input*Text_Vectorizer/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
/Text_Vectorizer/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
1Text_Vectorizer/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
1Text_Vectorizer/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
)Text_Vectorizer/StringSplit/strided_sliceStridedSlice3Text_Vectorizer/StringSplit/StringSplitV2:indices:08Text_Vectorizer/StringSplit/strided_slice/stack:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_1:output:0:Text_Vectorizer/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask{
1Text_Vectorizer/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3Text_Vectorizer/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3Text_Vectorizer/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+Text_Vectorizer/StringSplit/strided_slice_1StridedSlice1Text_Vectorizer/StringSplit/StringSplitV2:shape:0:Text_Vectorizer/StringSplit/strided_slice_1/stack:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_1:output:0<Text_Vectorizer/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
RText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast2Text_Vectorizer/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast4Text_Vectorizer/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
::���
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdeText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
`Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterdText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0iText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
[Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastbText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
\Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2cText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0eText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumXText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
dText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
^Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeVText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:����������
_Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountgText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0gText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumfText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
]Text_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
YText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
TText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2fText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0ZText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0bText_Vectorizer/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Htext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_table_handle2Text_Vectorizer/StringSplit/StringSplitV2:values:0Itext_vectorizer_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
#Text_Vectorizer/string_lookup/EqualEqual2Text_Vectorizer/StringSplit/StringSplitV2:values:0%text_vectorizer_string_lookup_equal_y*
T0*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/SelectV2SelectV2'Text_Vectorizer/string_lookup/Equal:z:0(text_vectorizer_string_lookup_selectv2_tDText_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
&Text_Vectorizer/string_lookup/IdentityIdentity/Text_Vectorizer/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������n
,Text_Vectorizer/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R }
$Text_Vectorizer/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������	       �
3Text_Vectorizer/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor-Text_Vectorizer/RaggedToTensor/Const:output:0/Text_Vectorizer/string_lookup/Identity:output:05Text_Vectorizer/RaggedToTensor/default_value:output:04Text_Vectorizer/StringSplit/strided_slice_1:output:02Text_Vectorizer/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������	*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
'Embedding_Layer/StatefulPartitionedCallStatefulPartitionedCall<Text_Vectorizer/RaggedToTensor/RaggedTensorToTensor:result:0embedding_layer_2553*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_2552�
&Hidden_Layer_1/StatefulPartitionedCallStatefulPartitionedCall0Embedding_Layer/StatefulPartitionedCall:output:0hidden_layer_1_2588hidden_layer_1_2590*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_2587�
&Hidden_Layer_2/StatefulPartitionedCallStatefulPartitionedCall/Hidden_Layer_1/StatefulPartitionedCall:output:0hidden_layer_2_2625hidden_layer_2_2627*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_2624�
$Output_Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_Layer_2/StatefulPartitionedCall:output:0output_layer_2662output_layer_2664*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_Output_Layer_layer_call_and_return_conditional_losses_2661�
IdentityIdentity-Output_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		�
NoOpNoOp(^Embedding_Layer/StatefulPartitionedCall'^Hidden_Layer_1/StatefulPartitionedCall'^Hidden_Layer_2/StatefulPartitionedCall%^Output_Layer/StatefulPartitionedCall<^Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 2R
'Embedding_Layer/StatefulPartitionedCall'Embedding_Layer/StatefulPartitionedCall2P
&Hidden_Layer_1/StatefulPartitionedCall&Hidden_Layer_1/StatefulPartitionedCall2P
&Hidden_Layer_2/StatefulPartitionedCall&Hidden_Layer_2/StatefulPartitionedCall2L
$Output_Layer/StatefulPartitionedCall$Output_Layer/StatefulPartitionedCall2z
;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2;Text_Vectorizer/string_lookup/None_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:���������
/
_user_specified_nameText_Vectorizer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
F__inference_Output_Layer_layer_call_and_return_conditional_losses_2661

inputs3
!tensordot_readvariableop_resource:
	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:
	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������	
�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������		r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������		Z
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:���������		d
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:���������		z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������	

 
_user_specified_nameinputs
�
�
7__inference_Embedding_Analysis_Model_layer_call_fn_2834
text_vectorizer_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	�
	unknown_4:
	unknown_5:
	unknown_6:

	unknown_7:

	unknown_8:
	
	unknown_9:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_vectorizer_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������		*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2809s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:���������
/
_user_specified_nameText_Vectorizer_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
.__inference_Embedding_Layer_layer_call_fn_3438

inputs	
unknown:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������	*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_2552s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������	: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_3447

inputs	(
embedding_lookup_3441:	�
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_3441inputs*
Tindices0	*(
_class
loc:@embedding_lookup/3441*+
_output_shapes
:���������	*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/3441*+
_output_shapes
:���������	�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������	w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������	Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������	: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
S
Text_Vectorizer_input:
'serving_default_Text_Vectorizer_input:0���������D
Output_Layer4
StatefulPartitionedCall:0���������		tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	vectorize
	embedding
hidden_layer1
hidden_layer2
output_layer
	optimizer

signatures"
_tf_keras_sequential
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
Q
1
2
 3
'4
(5
/6
07"
trackable_list_wrapper
Q
0
1
 2
'3
(4
/5
06"
trackable_list_wrapper
 "
trackable_list_wrapper
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
6trace_0
7trace_1
8trace_2
9trace_32�
7__inference_Embedding_Analysis_Model_layer_call_fn_2834
7__inference_Embedding_Analysis_Model_layer_call_fn_2930
7__inference_Embedding_Analysis_Model_layer_call_fn_3130
7__inference_Embedding_Analysis_Model_layer_call_fn_3157�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z6trace_0z7trace_1z8trace_2z9trace_3
�
:trace_0
;trace_1
<trace_2
=trace_32�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2668
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2737
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_3294
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_3431�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z:trace_0z;trace_1z<trace_2z=trace_3
�
>	capture_1
?	capture_2
@	capture_3B�
__inference__wrapped_model_2492Text_Vectorizer_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
�
A
_variables
B_iterations
C_learning_rate
D_index_dict
E
_momentums
F_velocities
G_update_step_xla"
experimentalOptimizer
,
Hserving_default"
signature_map
"
_generic_user_object
L
I	keras_api
Jlookup_table
Ktoken_counts"
_tf_keras_layer
�
Ltrace_02�
__inference_adapt_step_3103�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Rtrace_02�
.__inference_Embedding_Layer_layer_call_fn_3438�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0
�
Strace_02�
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_3447�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0
-:+	�2Embedding_Layer/embeddings
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ytrace_02�
-__inference_Hidden_Layer_1_layer_call_fn_3456�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zYtrace_0
�
Ztrace_02�
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_3487�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0
':%2Hidden_Layer_1/kernel
!:2Hidden_Layer_1/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
`trace_02�
-__inference_Hidden_Layer_2_layer_call_fn_3496�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0
�
atrace_02�
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_3527�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
':%
2Hidden_Layer_2/kernel
!:
2Hidden_Layer_2/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
gtrace_02�
+__inference_Output_Layer_layer_call_fn_3536�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zgtrace_0
�
htrace_02�
F__inference_Output_Layer_layer_call_and_return_conditional_losses_3567�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
%:#
	2Output_Layer/kernel
:	2Output_Layer/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
>	capture_1
?	capture_2
@	capture_3B�
7__inference_Embedding_Analysis_Model_layer_call_fn_2834Text_Vectorizer_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
�
>	capture_1
?	capture_2
@	capture_3B�
7__inference_Embedding_Analysis_Model_layer_call_fn_2930Text_Vectorizer_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
�
>	capture_1
?	capture_2
@	capture_3B�
7__inference_Embedding_Analysis_Model_layer_call_fn_3130inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
�
>	capture_1
?	capture_2
@	capture_3B�
7__inference_Embedding_Analysis_Model_layer_call_fn_3157inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
�
>	capture_1
?	capture_2
@	capture_3B�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2668Text_Vectorizer_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
�
>	capture_1
?	capture_2
@	capture_3B�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2737Text_Vectorizer_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
�
>	capture_1
?	capture_2
@	capture_3B�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_3294inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
�
>	capture_1
?	capture_2
@	capture_3B�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_3431inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
�
B0
k1
l2
m3
n4
o5
p6
q7
r8
s9
t10
u11
v12
w13
x14"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
Q
k0
m1
o2
q3
s4
u5
w6"
trackable_list_wrapper
Q
l0
n1
p2
r3
t4
v5
x6"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
>	capture_1
?	capture_2
@	capture_3B�
"__inference_signature_wrapper_3055Text_Vectorizer_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>	capture_1z?	capture_2z@	capture_3
"
_generic_user_object
f
y_initializer
z_create_resource
{_initialize
|_destroy_resourceR jtf.StaticHashTable
Q
}_create_resource
~_initialize
_destroy_resourceR Z
table��
�
�	capture_1B�
__inference_adapt_step_3103iterator"�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_Embedding_Layer_layer_call_fn_3438inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_3447inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_Hidden_Layer_1_layer_call_fn_3456inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_3487inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_Hidden_Layer_2_layer_call_fn_3496inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_3527inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_Output_Layer_layer_call_fn_3536inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_Output_Layer_layer_call_and_return_conditional_losses_3567inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
2:0	�2!Adam/m/Embedding_Layer/embeddings
2:0	�2!Adam/v/Embedding_Layer/embeddings
,:*2Adam/m/Hidden_Layer_1/kernel
,:*2Adam/v/Hidden_Layer_1/kernel
&:$2Adam/m/Hidden_Layer_1/bias
&:$2Adam/v/Hidden_Layer_1/bias
,:*
2Adam/m/Hidden_Layer_2/kernel
,:*
2Adam/v/Hidden_Layer_2/kernel
&:$
2Adam/m/Hidden_Layer_2/bias
&:$
2Adam/v/Hidden_Layer_2/bias
*:(
	2Adam/m/Output_Layer/kernel
*:(
	2Adam/v/Output_Layer/kernel
$:"	2Adam/m/Output_Layer/bias
$:"	2Adam/v/Output_Layer/bias
"
_generic_user_object
�
�trace_02�
__inference__creator_3572�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__initializer_3580�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__destroyer_3585�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__creator_3590�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__initializer_3595�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference__destroyer_3600�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
!J	
Const_2jtf.TrackableConstant
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
�B�
__inference__creator_3572"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�
�	capture_1
�	capture_2B�
__inference__initializer_3580"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�	capture_1z�	capture_2
�B�
__inference__destroyer_3585"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__creator_3590"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__initializer_3595"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__destroyer_3600"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�B�
__inference_save_fn_3619checkpoint_key"�
���
FullArgSpec
args�
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�	
� 
�B�
__inference_restore_fn_3627restored_tensors_0restored_tensors_1"�
���
FullArgSpec7
args/�,
jrestored_tensors_0
jrestored_tensors_1
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
	�	�
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2668�J>?@ '(/0B�?
8�5
+�(
Text_Vectorizer_input���������
p

 
� "0�-
&�#
tensor_0���������		
� �
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_2737�J>?@ '(/0B�?
8�5
+�(
Text_Vectorizer_input���������
p 

 
� "0�-
&�#
tensor_0���������		
� �
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_3294tJ>?@ '(/03�0
)�&
�
inputs���������
p

 
� "0�-
&�#
tensor_0���������		
� �
R__inference_Embedding_Analysis_Model_layer_call_and_return_conditional_losses_3431tJ>?@ '(/03�0
)�&
�
inputs���������
p 

 
� "0�-
&�#
tensor_0���������		
� �
7__inference_Embedding_Analysis_Model_layer_call_fn_2834xJ>?@ '(/0B�?
8�5
+�(
Text_Vectorizer_input���������
p

 
� "%�"
unknown���������		�
7__inference_Embedding_Analysis_Model_layer_call_fn_2930xJ>?@ '(/0B�?
8�5
+�(
Text_Vectorizer_input���������
p 

 
� "%�"
unknown���������		�
7__inference_Embedding_Analysis_Model_layer_call_fn_3130iJ>?@ '(/03�0
)�&
�
inputs���������
p

 
� "%�"
unknown���������		�
7__inference_Embedding_Analysis_Model_layer_call_fn_3157iJ>?@ '(/03�0
)�&
�
inputs���������
p 

 
� "%�"
unknown���������		�
I__inference_Embedding_Layer_layer_call_and_return_conditional_losses_3447f/�,
%�"
 �
inputs���������		
� "0�-
&�#
tensor_0���������	
� �
.__inference_Embedding_Layer_layer_call_fn_3438[/�,
%�"
 �
inputs���������		
� "%�"
unknown���������	�
H__inference_Hidden_Layer_1_layer_call_and_return_conditional_losses_3487k 3�0
)�&
$�!
inputs���������	
� "0�-
&�#
tensor_0���������	
� �
-__inference_Hidden_Layer_1_layer_call_fn_3456` 3�0
)�&
$�!
inputs���������	
� "%�"
unknown���������	�
H__inference_Hidden_Layer_2_layer_call_and_return_conditional_losses_3527k'(3�0
)�&
$�!
inputs���������	
� "0�-
&�#
tensor_0���������	

� �
-__inference_Hidden_Layer_2_layer_call_fn_3496`'(3�0
)�&
$�!
inputs���������	
� "%�"
unknown���������	
�
F__inference_Output_Layer_layer_call_and_return_conditional_losses_3567k/03�0
)�&
$�!
inputs���������	

� "0�-
&�#
tensor_0���������		
� �
+__inference_Output_Layer_layer_call_fn_3536`/03�0
)�&
$�!
inputs���������	

� "%�"
unknown���������		>
__inference__creator_3572!�

� 
� "�
unknown >
__inference__creator_3590!�

� 
� "�
unknown @
__inference__destroyer_3585!�

� 
� "�
unknown @
__inference__destroyer_3600!�

� 
� "�
unknown I
__inference__initializer_3580(J���

� 
� "�
unknown B
__inference__initializer_3595!�

� 
� "�
unknown �
__inference__wrapped_model_2492�J>?@ '(/0:�7
0�-
+�(
Text_Vectorizer_input���������
� "?�<
:
Output_Layer*�'
output_layer���������		i
__inference_adapt_step_3103JK�?�<
5�2
0�-�
����������IteratorSpec 
� "
 �
__inference_restore_fn_3627bKK�H
A�>
�
restored_tensors_0
�
restored_tensors_1	
� "�
unknown �
__inference_save_fn_3619�K&�#
�
�
checkpoint_key 
� "���
u�r

name�
tensor_0_name 
*

slice_spec�
tensor_0_slice_spec 
$
tensor�
tensor_0_tensor
u�r

name�
tensor_1_name 
*

slice_spec�
tensor_1_slice_spec 
$
tensor�
tensor_1_tensor	�
"__inference_signature_wrapper_3055�J>?@ '(/0S�P
� 
I�F
D
Text_Vectorizer_input+�(
text_vectorizer_input���������"?�<
:
Output_Layer*�'
output_layer���������		