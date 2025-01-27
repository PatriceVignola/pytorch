# Owner(s): ["module: inductor"]

from torch._inductor.test_case import run_tests, TestCase

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_cuda

try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False


class TestSplitCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        primals_1,
        primals_2,
        primals_3,
        primals_4,
        primals_5,
        primals_6,
        primals_7,
        primals_8,
        primals_9,
        primals_10,
        primals_11,
    ):
        cat = torch.ops.aten.cat.default(
            [primals_6, primals_5, primals_4, primals_3, primals_2, primals_1], 1
        )
        primals_6 = primals_5 = primals_4 = primals_3 = primals_2 = primals_1 = None
        split = torch.ops.aten.split.Tensor(cat, 128, 1)
        cat = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]
        getitem_6 = split[6]
        getitem_7 = split[7]
        getitem_8 = split[8]
        getitem_9 = split[9]
        getitem_10 = split[10]
        getitem_11 = split[11]
        getitem_12 = split[12]
        getitem_13 = split[13]
        getitem_14 = split[14]
        getitem_15 = split[15]
        getitem_16 = split[16]
        getitem_17 = split[17]
        getitem_18 = split[18]
        getitem_19 = split[19]
        getitem_20 = split[20]
        getitem_21 = split[21]
        getitem_22 = split[22]
        getitem_23 = split[23]
        getitem_24 = split[24]
        getitem_25 = split[25]
        getitem_26 = split[26]
        getitem_27 = split[27]
        getitem_28 = split[28]
        getitem_29 = split[29]
        getitem_30 = split[30]
        getitem_31 = split[31]
        getitem_32 = split[32]
        getitem_33 = split[33]
        getitem_34 = split[34]
        getitem_35 = split[35]
        getitem_36 = split[36]
        getitem_37 = split[37]
        getitem_38 = split[38]
        getitem_39 = split[39]
        getitem_40 = split[40]
        getitem_41 = split[41]
        getitem_42 = split[42]
        getitem_43 = split[43]
        getitem_44 = split[44]
        getitem_45 = split[45]
        getitem_46 = split[46]
        getitem_47 = split[47]
        getitem_48 = split[48]
        getitem_49 = split[49]
        getitem_50 = split[50]
        getitem_51 = split[51]
        getitem_52 = split[52]
        getitem_53 = split[53]
        getitem_54 = split[54]
        getitem_55 = split[55]
        getitem_56 = split[56]
        getitem_57 = split[57]
        getitem_58 = split[58]
        getitem_59 = split[59]
        getitem_60 = split[60]
        getitem_61 = split[61]
        getitem_62 = split[62]
        getitem_63 = split[63]
        getitem_64 = split[64]
        getitem_65 = split[65]
        getitem_66 = split[66]
        getitem_67 = split[67]
        getitem_68 = split[68]
        getitem_69 = split[69]
        getitem_70 = split[70]
        getitem_71 = split[71]
        getitem_72 = split[72]
        getitem_73 = split[73]
        getitem_74 = split[74]
        getitem_75 = split[75]
        getitem_76 = split[76]
        getitem_77 = split[77]
        getitem_78 = split[78]
        getitem_79 = split[79]
        getitem_80 = split[80]
        getitem_81 = split[81]
        getitem_82 = split[82]
        getitem_83 = split[83]
        getitem_84 = split[84]
        getitem_85 = split[85]
        getitem_86 = split[86]
        getitem_87 = split[87]
        getitem_88 = split[88]
        getitem_89 = split[89]
        getitem_90 = split[90]
        getitem_91 = split[91]
        getitem_92 = split[92]
        getitem_93 = split[93]
        getitem_94 = split[94]
        getitem_95 = split[95]
        getitem_96 = split[96]
        getitem_97 = split[97]
        getitem_98 = split[98]
        getitem_99 = split[99]
        getitem_100 = split[100]
        getitem_101 = split[101]
        getitem_102 = split[102]
        getitem_103 = split[103]
        getitem_104 = split[104]
        getitem_105 = split[105]
        getitem_106 = split[106]
        getitem_107 = split[107]
        getitem_108 = split[108]
        getitem_109 = split[109]
        getitem_110 = split[110]
        getitem_111 = split[111]
        getitem_112 = split[112]
        getitem_113 = split[113]
        getitem_114 = split[114]
        getitem_115 = split[115]
        getitem_116 = split[116]
        getitem_117 = split[117]
        getitem_118 = split[118]
        getitem_119 = split[119]
        getitem_120 = split[120]
        getitem_121 = split[121]
        getitem_122 = split[122]
        getitem_123 = split[123]
        getitem_124 = split[124]
        getitem_125 = split[125]
        getitem_126 = split[126]
        getitem_127 = split[127]
        getitem_128 = split[128]
        getitem_129 = split[129]
        getitem_130 = split[130]
        getitem_131 = split[131]
        getitem_132 = split[132]
        getitem_133 = split[133]
        getitem_134 = split[134]
        getitem_135 = split[135]
        getitem_136 = split[136]
        getitem_137 = split[137]
        getitem_138 = split[138]
        getitem_139 = split[139]
        getitem_140 = split[140]
        getitem_141 = split[141]
        getitem_142 = split[142]
        getitem_143 = split[143]
        getitem_144 = split[144]
        getitem_145 = split[145]
        getitem_146 = split[146]
        getitem_147 = split[147]
        getitem_148 = split[148]
        getitem_149 = split[149]
        getitem_150 = split[150]
        getitem_151 = split[151]
        getitem_152 = split[152]
        getitem_153 = split[153]
        getitem_154 = split[154]
        getitem_155 = split[155]
        getitem_156 = split[156]
        getitem_157 = split[157]
        getitem_158 = split[158]
        getitem_159 = split[159]
        getitem_160 = split[160]
        getitem_161 = split[161]
        getitem_162 = split[162]
        getitem_163 = split[163]
        getitem_164 = split[164]
        getitem_165 = split[165]
        getitem_166 = split[166]
        getitem_167 = split[167]
        getitem_168 = split[168]
        getitem_169 = split[169]
        getitem_170 = split[170]
        getitem_171 = split[171]
        getitem_172 = split[172]
        getitem_173 = split[173]
        getitem_174 = split[174]
        getitem_175 = split[175]
        getitem_176 = split[176]
        getitem_177 = split[177]
        getitem_178 = split[178]
        getitem_179 = split[179]
        getitem_180 = split[180]
        getitem_181 = split[181]
        getitem_182 = split[182]
        getitem_183 = split[183]
        getitem_184 = split[184]
        getitem_185 = split[185]
        getitem_186 = split[186]
        getitem_187 = split[187]
        getitem_188 = split[188]
        getitem_189 = split[189]
        getitem_190 = split[190]
        getitem_191 = split[191]
        getitem_192 = split[192]
        getitem_193 = split[193]
        getitem_194 = split[194]
        getitem_195 = split[195]
        getitem_196 = split[196]
        getitem_197 = split[197]
        getitem_198 = split[198]
        getitem_199 = split[199]
        getitem_200 = split[200]
        getitem_201 = split[201]
        getitem_202 = split[202]
        getitem_203 = split[203]
        getitem_204 = split[204]
        getitem_205 = split[205]
        getitem_206 = split[206]
        getitem_207 = split[207]
        getitem_208 = split[208]
        getitem_209 = split[209]
        getitem_210 = split[210]
        getitem_211 = split[211]
        getitem_212 = split[212]
        getitem_213 = split[213]
        getitem_214 = split[214]
        getitem_215 = split[215]
        getitem_216 = split[216]
        getitem_217 = split[217]
        getitem_218 = split[218]
        getitem_219 = split[219]
        getitem_220 = split[220]
        getitem_221 = split[221]
        getitem_222 = split[222]
        getitem_223 = split[223]
        getitem_224 = split[224]
        getitem_225 = split[225]
        getitem_226 = split[226]
        getitem_227 = split[227]
        getitem_228 = split[228]
        getitem_229 = split[229]
        getitem_230 = split[230]
        getitem_231 = split[231]
        getitem_232 = split[232]
        getitem_233 = split[233]
        getitem_234 = split[234]
        getitem_235 = split[235]
        getitem_236 = split[236]
        getitem_237 = split[237]
        getitem_238 = split[238]
        getitem_239 = split[239]
        getitem_240 = split[240]
        getitem_241 = split[241]
        getitem_242 = split[242]
        getitem_243 = split[243]
        getitem_244 = split[244]
        getitem_245 = split[245]
        getitem_246 = split[246]
        getitem_247 = split[247]
        getitem_248 = split[248]
        getitem_249 = split[249]
        getitem_250 = split[250]
        getitem_251 = split[251]
        getitem_252 = split[252]
        getitem_253 = split[253]
        getitem_254 = split[254]
        getitem_255 = split[255]
        getitem_256 = split[256]
        getitem_257 = split[257]
        getitem_258 = split[258]
        getitem_259 = split[259]
        getitem_260 = split[260]
        getitem_261 = split[261]
        getitem_262 = split[262]
        getitem_263 = split[263]
        getitem_264 = split[264]
        getitem_265 = split[265]
        getitem_266 = split[266]
        getitem_267 = split[267]
        getitem_268 = split[268]
        getitem_269 = split[269]
        getitem_270 = split[270]
        getitem_271 = split[271]
        getitem_272 = split[272]
        getitem_273 = split[273]
        getitem_274 = split[274]
        getitem_275 = split[275]
        getitem_276 = split[276]
        getitem_277 = split[277]
        getitem_278 = split[278]
        getitem_279 = split[279]
        getitem_280 = split[280]
        getitem_281 = split[281]
        getitem_282 = split[282]
        getitem_283 = split[283]
        getitem_284 = split[284]
        getitem_285 = split[285]
        getitem_286 = split[286]
        getitem_287 = split[287]
        getitem_288 = split[288]
        getitem_289 = split[289]
        getitem_290 = split[290]
        getitem_291 = split[291]
        getitem_292 = split[292]
        getitem_293 = split[293]
        getitem_294 = split[294]
        getitem_295 = split[295]
        getitem_296 = split[296]
        getitem_297 = split[297]
        getitem_298 = split[298]
        getitem_299 = split[299]
        getitem_300 = split[300]
        getitem_301 = split[301]
        getitem_302 = split[302]
        getitem_303 = split[303]
        getitem_304 = split[304]
        getitem_305 = split[305]
        getitem_306 = split[306]
        getitem_307 = split[307]
        getitem_308 = split[308]
        getitem_309 = split[309]
        getitem_310 = split[310]
        getitem_311 = split[311]
        getitem_312 = split[312]
        getitem_313 = split[313]
        getitem_314 = split[314]
        getitem_315 = split[315]
        getitem_316 = split[316]
        getitem_317 = split[317]
        getitem_318 = split[318]
        getitem_319 = split[319]
        getitem_320 = split[320]
        getitem_321 = split[321]
        getitem_322 = split[322]
        getitem_323 = split[323]
        getitem_324 = split[324]
        getitem_325 = split[325]
        getitem_326 = split[326]
        getitem_327 = split[327]
        getitem_328 = split[328]
        getitem_329 = split[329]
        getitem_330 = split[330]
        getitem_331 = split[331]
        getitem_332 = split[332]
        getitem_333 = split[333]
        getitem_334 = split[334]
        getitem_335 = split[335]
        getitem_336 = split[336]
        getitem_337 = split[337]
        getitem_338 = split[338]
        getitem_339 = split[339]
        getitem_340 = split[340]
        getitem_341 = split[341]
        getitem_342 = split[342]
        getitem_343 = split[343]
        getitem_344 = split[344]
        getitem_345 = split[345]
        getitem_346 = split[346]
        getitem_347 = split[347]
        getitem_348 = split[348]
        getitem_349 = split[349]
        getitem_350 = split[350]
        getitem_351 = split[351]
        getitem_352 = split[352]
        getitem_353 = split[353]
        getitem_354 = split[354]
        getitem_355 = split[355]
        getitem_356 = split[356]
        getitem_357 = split[357]
        getitem_358 = split[358]
        getitem_359 = split[359]
        getitem_360 = split[360]
        getitem_361 = split[361]
        getitem_362 = split[362]
        getitem_363 = split[363]
        getitem_364 = split[364]
        getitem_365 = split[365]
        getitem_366 = split[366]
        getitem_367 = split[367]
        getitem_368 = split[368]
        getitem_369 = split[369]
        getitem_370 = split[370]
        getitem_371 = split[371]
        getitem_372 = split[372]
        getitem_373 = split[373]
        getitem_374 = split[374]
        getitem_375 = split[375]
        getitem_376 = split[376]
        getitem_377 = split[377]
        getitem_378 = split[378]
        getitem_379 = split[379]
        getitem_380 = split[380]
        getitem_381 = split[381]
        getitem_382 = split[382]
        getitem_383 = split[383]
        getitem_384 = split[384]
        getitem_385 = split[385]
        getitem_386 = split[386]
        getitem_387 = split[387]
        getitem_388 = split[388]
        getitem_389 = split[389]
        getitem_390 = split[390]
        getitem_391 = split[391]
        getitem_392 = split[392]
        getitem_393 = split[393]
        getitem_394 = split[394]
        getitem_395 = split[395]
        getitem_396 = split[396]
        getitem_397 = split[397]
        getitem_398 = split[398]
        getitem_399 = split[399]
        getitem_400 = split[400]
        getitem_401 = split[401]
        getitem_402 = split[402]
        getitem_403 = split[403]
        getitem_404 = split[404]
        getitem_405 = split[405]
        getitem_406 = split[406]
        getitem_407 = split[407]
        getitem_408 = split[408]
        getitem_409 = split[409]
        getitem_410 = split[410]
        getitem_411 = split[411]
        getitem_412 = split[412]
        getitem_413 = split[413]
        getitem_414 = split[414]
        getitem_415 = split[415]
        getitem_416 = split[416]
        getitem_417 = split[417]
        getitem_418 = split[418]
        getitem_419 = split[419]
        getitem_420 = split[420]
        getitem_421 = split[421]
        getitem_422 = split[422]
        getitem_423 = split[423]
        getitem_424 = split[424]
        getitem_425 = split[425]
        getitem_426 = split[426]
        getitem_427 = split[427]
        getitem_428 = split[428]
        getitem_429 = split[429]
        getitem_430 = split[430]
        getitem_431 = split[431]
        getitem_432 = split[432]
        getitem_433 = split[433]
        getitem_434 = split[434]
        getitem_435 = split[435]
        getitem_436 = split[436]
        getitem_437 = split[437]
        getitem_438 = split[438]
        getitem_439 = split[439]
        getitem_440 = split[440]
        getitem_441 = split[441]
        getitem_442 = split[442]
        getitem_443 = split[443]
        getitem_444 = split[444]
        getitem_445 = split[445]
        getitem_446 = split[446]
        getitem_447 = split[447]
        getitem_448 = split[448]
        getitem_449 = split[449]
        getitem_450 = split[450]
        getitem_451 = split[451]
        getitem_452 = split[452]
        getitem_453 = split[453]
        getitem_454 = split[454]
        getitem_455 = split[455]
        getitem_456 = split[456]
        getitem_457 = split[457]
        getitem_458 = split[458]
        getitem_459 = split[459]
        getitem_460 = split[460]
        getitem_461 = split[461]
        getitem_462 = split[462]
        getitem_463 = split[463]
        getitem_464 = split[464]
        getitem_465 = split[465]
        getitem_466 = split[466]
        getitem_467 = split[467]
        getitem_468 = split[468]
        getitem_469 = split[469]
        getitem_470 = split[470]
        getitem_471 = split[471]
        getitem_472 = split[472]
        getitem_473 = split[473]
        getitem_474 = split[474]
        getitem_475 = split[475]
        getitem_476 = split[476]
        getitem_477 = split[477]
        getitem_478 = split[478]
        getitem_479 = split[479]
        getitem_480 = split[480]
        getitem_481 = split[481]
        getitem_482 = split[482]
        getitem_483 = split[483]
        getitem_484 = split[484]
        getitem_485 = split[485]
        getitem_486 = split[486]
        getitem_487 = split[487]
        getitem_488 = split[488]
        getitem_489 = split[489]
        split = None
        cat_1 = torch.ops.aten.cat.default(
            [
                getitem,
                getitem_1,
                getitem_2,
                getitem_3,
                getitem_4,
                getitem_5,
                getitem_6,
                getitem_7,
                getitem_8,
                getitem_9,
                getitem_10,
                getitem_11,
                getitem_12,
                getitem_13,
                getitem_14,
                getitem_15,
                getitem_16,
                getitem_17,
                getitem_18,
                getitem_19,
                getitem_20,
                getitem_21,
                getitem_22,
                getitem_23,
                getitem_24,
                getitem_25,
                getitem_26,
                getitem_27,
                getitem_28,
                getitem_29,
                getitem_30,
                getitem_31,
                getitem_32,
                getitem_33,
                getitem_34,
                getitem_35,
                getitem_36,
                getitem_37,
                getitem_38,
                getitem_39,
                getitem_40,
                getitem_41,
                getitem_42,
                getitem_43,
                getitem_44,
                getitem_45,
                getitem_46,
                getitem_47,
                getitem_48,
                getitem_49,
                getitem_50,
                getitem_51,
                getitem_52,
                getitem_53,
                getitem_54,
                getitem_55,
                getitem_56,
                getitem_57,
                getitem_58,
                getitem_59,
                getitem_60,
                getitem_61,
                getitem_62,
                getitem_63,
                getitem_64,
                getitem_65,
                getitem_66,
                getitem_67,
                getitem_68,
                getitem_69,
                getitem_70,
                getitem_71,
                getitem_72,
                getitem_73,
                getitem_74,
                getitem_75,
                getitem_76,
                getitem_77,
                getitem_78,
                getitem_79,
                getitem_80,
                getitem_81,
                getitem_82,
                getitem_83,
                getitem_84,
                getitem_85,
                getitem_86,
                getitem_87,
                getitem_88,
                getitem_89,
                getitem_90,
                getitem_91,
                getitem_92,
                getitem_93,
                getitem_94,
                getitem_95,
                getitem_96,
                getitem_97,
                getitem_98,
                getitem_99,
                getitem_100,
                getitem_101,
                getitem_102,
                getitem_103,
                getitem_104,
                getitem_105,
                getitem_106,
                getitem_107,
                getitem_108,
                getitem_109,
                getitem_110,
                getitem_111,
                getitem_112,
                getitem_113,
                getitem_114,
                getitem_115,
                getitem_116,
                getitem_117,
                getitem_118,
                getitem_119,
                getitem_120,
                getitem_121,
                getitem_122,
                getitem_123,
                getitem_124,
                getitem_125,
                getitem_126,
                getitem_127,
                getitem_128,
                getitem_129,
                getitem_130,
                getitem_131,
                getitem_132,
                getitem_133,
                getitem_134,
                getitem_135,
                getitem_136,
                getitem_137,
                getitem_138,
                getitem_139,
                getitem_140,
                getitem_141,
                getitem_142,
                getitem_143,
                getitem_144,
                getitem_145,
                getitem_146,
                getitem_147,
                getitem_148,
                getitem_149,
                getitem_150,
                getitem_151,
                getitem_152,
                getitem_153,
                getitem_154,
                getitem_155,
                getitem_156,
                getitem_157,
                getitem_158,
                getitem_159,
                getitem_160,
                getitem_161,
                getitem_162,
                getitem_163,
                getitem_164,
                getitem_165,
                getitem_166,
                getitem_167,
                getitem_168,
                getitem_169,
                getitem_170,
                getitem_171,
                getitem_172,
                getitem_173,
                getitem_174,
                getitem_175,
                getitem_176,
                getitem_177,
                getitem_178,
                getitem_179,
                getitem_180,
                getitem_181,
                getitem_182,
                getitem_183,
                getitem_184,
                getitem_185,
                getitem_186,
                getitem_187,
                getitem_188,
                getitem_189,
                getitem_190,
                getitem_191,
                getitem_192,
                getitem_193,
                getitem_194,
                getitem_195,
                getitem_196,
                getitem_197,
                getitem_198,
                getitem_199,
                getitem_200,
                getitem_201,
                getitem_202,
                getitem_203,
                getitem_204,
                getitem_205,
                getitem_206,
                getitem_207,
                getitem_208,
                getitem_209,
                getitem_210,
                getitem_211,
                getitem_212,
                getitem_213,
                getitem_214,
                getitem_215,
                getitem_216,
                getitem_217,
                getitem_218,
                getitem_219,
                getitem_220,
                getitem_221,
                getitem_222,
                getitem_223,
                getitem_224,
                getitem_225,
                getitem_226,
                getitem_227,
                getitem_228,
                getitem_229,
                getitem_230,
                getitem_231,
                getitem_232,
                getitem_233,
                getitem_234,
                getitem_235,
                getitem_236,
                getitem_237,
                getitem_238,
                getitem_239,
                getitem_240,
                getitem_241,
                getitem_242,
                getitem_243,
                getitem_244,
                getitem_245,
                getitem_246,
                getitem_247,
                getitem_248,
                getitem_249,
                getitem_250,
                getitem_251,
                getitem_252,
                getitem_253,
                getitem_254,
                getitem_255,
                getitem_256,
                getitem_257,
                getitem_258,
                getitem_259,
                getitem_260,
                getitem_261,
                getitem_262,
                getitem_263,
                getitem_264,
                getitem_265,
                getitem_266,
                getitem_267,
                getitem_268,
                getitem_269,
                getitem_270,
                getitem_271,
                getitem_272,
                getitem_273,
                getitem_274,
                getitem_275,
                getitem_276,
                getitem_277,
                getitem_278,
                getitem_279,
                getitem_280,
                getitem_281,
                getitem_282,
                getitem_283,
                getitem_284,
                getitem_285,
                getitem_286,
                getitem_287,
                getitem_288,
                getitem_289,
                getitem_290,
                getitem_291,
                getitem_292,
                getitem_293,
                getitem_294,
                getitem_295,
                getitem_296,
                getitem_297,
                getitem_298,
                getitem_299,
                getitem_300,
                getitem_301,
                getitem_302,
                getitem_303,
                getitem_304,
                getitem_305,
                getitem_306,
                getitem_307,
                getitem_308,
                getitem_309,
                getitem_310,
                getitem_311,
                getitem_312,
                getitem_313,
                getitem_314,
                getitem_315,
                getitem_316,
                getitem_317,
                getitem_318,
                getitem_319,
                getitem_320,
                getitem_321,
                getitem_322,
                getitem_323,
                getitem_324,
                getitem_325,
                getitem_326,
                getitem_327,
                getitem_328,
                getitem_329,
                getitem_330,
                getitem_331,
                getitem_332,
                getitem_333,
                getitem_334,
                getitem_335,
                getitem_336,
                getitem_337,
                getitem_338,
                getitem_339,
                getitem_340,
                getitem_341,
                getitem_342,
                getitem_343,
                getitem_344,
                getitem_345,
                getitem_346,
                getitem_347,
                getitem_348,
                getitem_349,
                getitem_350,
                getitem_351,
                getitem_352,
                getitem_353,
                getitem_354,
                getitem_355,
                getitem_356,
                getitem_357,
                getitem_358,
                getitem_359,
                getitem_360,
                getitem_361,
                getitem_362,
                getitem_363,
                getitem_364,
                getitem_365,
                getitem_366,
                getitem_367,
                getitem_368,
                getitem_369,
                getitem_370,
                getitem_371,
                getitem_372,
                getitem_373,
                getitem_374,
                getitem_375,
                getitem_376,
                getitem_377,
                getitem_378,
                getitem_379,
                getitem_380,
                getitem_381,
                getitem_382,
                getitem_383,
                getitem_384,
                getitem_385,
                getitem_386,
                getitem_387,
                getitem_388,
                getitem_389,
                getitem_390,
                getitem_391,
                getitem_392,
                getitem_393,
                getitem_394,
                getitem_395,
                getitem_396,
                getitem_397,
                getitem_398,
                getitem_399,
                getitem_400,
                getitem_401,
                getitem_402,
                getitem_403,
                getitem_404,
                getitem_405,
                getitem_406,
                getitem_407,
                getitem_408,
                getitem_409,
                getitem_410,
                getitem_411,
                getitem_412,
                getitem_413,
                getitem_414,
                getitem_415,
                getitem_416,
                getitem_417,
                getitem_418,
                getitem_419,
                getitem_420,
                getitem_421,
                getitem_422,
                getitem_423,
                getitem_424,
                getitem_425,
                getitem_426,
                getitem_427,
                getitem_428,
                getitem_429,
                getitem_430,
                getitem_431,
                getitem_432,
                getitem_433,
                getitem_434,
                getitem_435,
                getitem_436,
                getitem_437,
                getitem_438,
                getitem_439,
                getitem_440,
                getitem_441,
                getitem_442,
                getitem_443,
                getitem_444,
                getitem_445,
                getitem_446,
                getitem_447,
                getitem_448,
                getitem_449,
                getitem_450,
                getitem_451,
                getitem_452,
                getitem_453,
                getitem_454,
                getitem_455,
                getitem_456,
                getitem_457,
                getitem_458,
                getitem_459,
                getitem_460,
                getitem_461,
                getitem_462,
                getitem_463,
                getitem_464,
                getitem_465,
                getitem_466,
                getitem_467,
                getitem_468,
                getitem_469,
                getitem_470,
                getitem_471,
                getitem_472,
                getitem_473,
                getitem_474,
                getitem_475,
                getitem_476,
                getitem_477,
                getitem_478,
                getitem_479,
                getitem_480,
                getitem_481,
                getitem_482,
                getitem_483,
                getitem_484,
                getitem_485,
                getitem_486,
                getitem_487,
                getitem_488,
                getitem_489,
            ],
            1,
        )
        getitem = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = (
            getitem_6
        ) = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = getitem_12 = (
            getitem_13
        ) = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = (
            getitem_19
        ) = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = (
            getitem_25
        ) = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = (
            getitem_31
        ) = getitem_32 = getitem_33 = getitem_34 = getitem_35 = getitem_36 = (
            getitem_37
        ) = getitem_38 = getitem_39 = getitem_40 = getitem_41 = getitem_42 = (
            getitem_43
        ) = getitem_44 = getitem_45 = getitem_46 = getitem_47 = getitem_48 = (
            getitem_49
        ) = getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = (
            getitem_55
        ) = getitem_56 = getitem_57 = getitem_58 = getitem_59 = getitem_60 = (
            getitem_61
        ) = getitem_62 = getitem_63 = getitem_64 = getitem_65 = getitem_66 = (
            getitem_67
        ) = getitem_68 = getitem_69 = getitem_70 = getitem_71 = getitem_72 = (
            getitem_73
        ) = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = (
            getitem_79
        ) = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = (
            getitem_85
        ) = getitem_86 = getitem_87 = getitem_88 = getitem_89 = getitem_90 = (
            getitem_91
        ) = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = (
            getitem_97
        ) = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = (
            getitem_103
        ) = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = (
            getitem_109
        ) = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = (
            getitem_115
        ) = getitem_116 = getitem_117 = getitem_118 = getitem_119 = getitem_120 = (
            getitem_121
        ) = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = (
            getitem_127
        ) = getitem_128 = getitem_129 = getitem_130 = getitem_131 = getitem_132 = (
            getitem_133
        ) = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = (
            getitem_139
        ) = getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = (
            getitem_145
        ) = getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = (
            getitem_151
        ) = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = (
            getitem_157
        ) = getitem_158 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = (
            getitem_163
        ) = getitem_164 = getitem_165 = getitem_166 = getitem_167 = getitem_168 = (
            getitem_169
        ) = getitem_170 = getitem_171 = getitem_172 = getitem_173 = getitem_174 = (
            getitem_175
        ) = getitem_176 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = (
            getitem_181
        ) = getitem_182 = getitem_183 = getitem_184 = getitem_185 = getitem_186 = (
            getitem_187
        ) = getitem_188 = getitem_189 = getitem_190 = getitem_191 = getitem_192 = (
            getitem_193
        ) = getitem_194 = getitem_195 = getitem_196 = getitem_197 = getitem_198 = (
            getitem_199
        ) = getitem_200 = getitem_201 = getitem_202 = getitem_203 = getitem_204 = (
            getitem_205
        ) = getitem_206 = getitem_207 = getitem_208 = getitem_209 = getitem_210 = (
            getitem_211
        ) = getitem_212 = getitem_213 = getitem_214 = getitem_215 = getitem_216 = (
            getitem_217
        ) = getitem_218 = getitem_219 = getitem_220 = getitem_221 = getitem_222 = (
            getitem_223
        ) = getitem_224 = getitem_225 = getitem_226 = getitem_227 = getitem_228 = (
            getitem_229
        ) = getitem_230 = getitem_231 = getitem_232 = getitem_233 = getitem_234 = (
            getitem_235
        ) = getitem_236 = getitem_237 = getitem_238 = getitem_239 = getitem_240 = (
            getitem_241
        ) = getitem_242 = getitem_243 = getitem_244 = getitem_245 = getitem_246 = (
            getitem_247
        ) = getitem_248 = getitem_249 = getitem_250 = getitem_251 = getitem_252 = (
            getitem_253
        ) = getitem_254 = getitem_255 = getitem_256 = getitem_257 = getitem_258 = (
            getitem_259
        ) = getitem_260 = getitem_261 = getitem_262 = getitem_263 = getitem_264 = (
            getitem_265
        ) = getitem_266 = getitem_267 = getitem_268 = getitem_269 = getitem_270 = (
            getitem_271
        ) = getitem_272 = getitem_273 = getitem_274 = getitem_275 = getitem_276 = (
            getitem_277
        ) = getitem_278 = getitem_279 = getitem_280 = getitem_281 = getitem_282 = (
            getitem_283
        ) = getitem_284 = getitem_285 = getitem_286 = getitem_287 = getitem_288 = (
            getitem_289
        ) = getitem_290 = getitem_291 = getitem_292 = getitem_293 = getitem_294 = (
            getitem_295
        ) = getitem_296 = getitem_297 = getitem_298 = getitem_299 = getitem_300 = (
            getitem_301
        ) = getitem_302 = getitem_303 = getitem_304 = getitem_305 = getitem_306 = (
            getitem_307
        ) = getitem_308 = getitem_309 = getitem_310 = getitem_311 = getitem_312 = (
            getitem_313
        ) = getitem_314 = getitem_315 = getitem_316 = getitem_317 = getitem_318 = (
            getitem_319
        ) = getitem_320 = getitem_321 = getitem_322 = getitem_323 = getitem_324 = (
            getitem_325
        ) = getitem_326 = getitem_327 = getitem_328 = getitem_329 = getitem_330 = (
            getitem_331
        ) = getitem_332 = getitem_333 = getitem_334 = getitem_335 = getitem_336 = (
            getitem_337
        ) = getitem_338 = getitem_339 = getitem_340 = getitem_341 = getitem_342 = (
            getitem_343
        ) = getitem_344 = getitem_345 = getitem_346 = getitem_347 = getitem_348 = (
            getitem_349
        ) = getitem_350 = getitem_351 = getitem_352 = getitem_353 = getitem_354 = (
            getitem_355
        ) = getitem_356 = getitem_357 = getitem_358 = getitem_359 = getitem_360 = (
            getitem_361
        ) = getitem_362 = getitem_363 = getitem_364 = getitem_365 = getitem_366 = (
            getitem_367
        ) = getitem_368 = getitem_369 = getitem_370 = getitem_371 = getitem_372 = (
            getitem_373
        ) = getitem_374 = getitem_375 = getitem_376 = getitem_377 = getitem_378 = (
            getitem_379
        ) = getitem_380 = getitem_381 = getitem_382 = getitem_383 = getitem_384 = (
            getitem_385
        ) = getitem_386 = getitem_387 = getitem_388 = getitem_389 = getitem_390 = (
            getitem_391
        ) = getitem_392 = getitem_393 = getitem_394 = getitem_395 = getitem_396 = (
            getitem_397
        ) = getitem_398 = getitem_399 = getitem_400 = getitem_401 = getitem_402 = (
            getitem_403
        ) = getitem_404 = getitem_405 = getitem_406 = getitem_407 = getitem_408 = (
            getitem_409
        ) = getitem_410 = getitem_411 = getitem_412 = getitem_413 = getitem_414 = (
            getitem_415
        ) = getitem_416 = getitem_417 = getitem_418 = getitem_419 = getitem_420 = (
            getitem_421
        ) = getitem_422 = getitem_423 = getitem_424 = getitem_425 = getitem_426 = (
            getitem_427
        ) = getitem_428 = getitem_429 = getitem_430 = getitem_431 = getitem_432 = (
            getitem_433
        ) = getitem_434 = getitem_435 = getitem_436 = getitem_437 = getitem_438 = (
            getitem_439
        ) = getitem_440 = getitem_441 = getitem_442 = getitem_443 = getitem_444 = (
            getitem_445
        ) = getitem_446 = getitem_447 = getitem_448 = getitem_449 = getitem_450 = (
            getitem_451
        ) = getitem_452 = getitem_453 = getitem_454 = getitem_455 = getitem_456 = (
            getitem_457
        ) = getitem_458 = getitem_459 = getitem_460 = getitem_461 = getitem_462 = (
            getitem_463
        ) = getitem_464 = getitem_465 = getitem_466 = getitem_467 = getitem_468 = (
            getitem_469
        ) = getitem_470 = getitem_471 = getitem_472 = getitem_473 = getitem_474 = (
            getitem_475
        ) = getitem_476 = getitem_477 = getitem_478 = getitem_479 = getitem_480 = (
            getitem_481
        ) = getitem_482 = getitem_483 = getitem_484 = getitem_485 = getitem_486 = (
            getitem_487
        ) = getitem_488 = getitem_489 = None
        view = torch.ops.aten.view.default(cat_1, [1024, 490, 128])
        cat_1 = None
        add = torch.ops.aten.add.Tensor(primals_8, primals_9)
        primals_8 = primals_9 = None
        view_1 = torch.ops.aten.view.default(primals_11, [-1, 1, 1, 800])
        primals_11 = None
        return (add, view, view_1, primals_7)

class TestSplitCatAten(TestCase):
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict.keys():
            key2 = "_orig_mod." + key1
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            if not torch.allclose(ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol):
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    @requires_cuda
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "split_cat_aten_pass": {},
        },
    )
    def test_split_cat_post_grad(self):
        counters.clear()
        inputs = [
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 3712, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 4608, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 20096, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 1664, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 32512, device=torch.device(device=GPU_TYPE)),
            torch.sym_int(1024),
            torch.randn(1024, 800, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 800, 128, device=torch.device(device=GPU_TYPE)),
            torch.sym_int(1024),
            torch.randint(0, 2, (1024, 800), device=torch.device(device=GPU_TYPE)),
        ]
        module = TestSplitCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 3)
        self.assertEqual(counters["inductor"]["split_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

if __name__ == "__main__":
    run_tests()
