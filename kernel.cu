#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdint.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
__host__ __device__ static void print_hex(const uint8_t* data, int len)
{
    for (int i = 0; i < len; i++) {
        printf("%02X", data[i]);
        if (i % 16 == 15)printf("\n");
    }
    printf("\n");
}
typedef struct lea_key_st
{
    unsigned int rk[192];
    unsigned int round;
} LEA_KEY;
typedef struct lea_gcm_ctx
{
    //uint8_t sub_h[4][16];
    uint8_t h[256][16];
    //uint8_t sub_GHASH[256];
    uint8_t ctr[16];
    uint8_t ek0[16];
    uint8_t tbl[16];   /* tag block */
    uint8_t yn[16];   /* last encrypted block */
    LEA_KEY key;

    int yn_used, aad_len, ct_len;
    int ghash_block;
    int is_encrypt;
} LEA_GCM_CTX;
__host__ __device__ unsigned int ROL(unsigned int u, int k) {
    u = (u << k) | (u >> (32 - k));
    return u;
}
__host__ __device__  unsigned int ROR(unsigned int u, int k) {
    u = (u >> k) | (u << (32 - k));
    return u;
}
__constant__  const unsigned int delta[8][36] = {
   { 0xc3efe9db, 0x87dfd3b7, 0x0fbfa76f, 0x1f7f4ede, 0x3efe9dbc, 0x7dfd3b78, 0xfbfa76f0, 0xf7f4ede1,
   0xefe9dbc3, 0xdfd3b787, 0xbfa76f0f, 0x7f4ede1f, 0xfe9dbc3e, 0xfd3b787d, 0xfa76f0fb, 0xf4ede1f7,
   0xe9dbc3ef, 0xd3b787df, 0xa76f0fbf, 0x4ede1f7f, 0x9dbc3efe, 0x3b787dfd, 0x76f0fbfa, 0xede1f7f4,
   0xdbc3efe9, 0xb787dfd3, 0x6f0fbfa7, 0xde1f7f4e, 0xbc3efe9d, 0x787dfd3b, 0xf0fbfa76, 0xe1f7f4eD,
   0xc3efe9db, 0x87dfd3b7, 0x0fbfa76f, 0x1f7f4ede },
   { 0x44626b02, 0x88c4d604, 0x1189ac09, 0x23135812, 0x4626b024, 0x8c4d6048, 0x189ac091, 0x31358122,
   0x626b0244, 0xc4d60488, 0x89ac0911, 0x13581223, 0x26b02446, 0x4d60488c, 0x9ac09118, 0x35812231,
   0x6b024462, 0xd60488c4, 0xac091189, 0x58122313, 0xb0244626, 0x60488c4d, 0xc091189a, 0x81223135,
   0x0244626b, 0x0488c4d6, 0x091189ac, 0x12231358, 0x244626b0, 0x488c4d60, 0x91189ac0, 0x22313581,
   0x44626b02, 0x88c4d604, 0x1189ac09, 0x23135812 },
   { 0x79e27c8a, 0xf3c4f914, 0xe789f229, 0xcf13e453, 0x9e27c8a7, 0x3c4f914f, 0x789f229e, 0xf13e453c,
   0xe27c8a79, 0xc4f914f3, 0x89f229e7, 0x13e453cf, 0x27c8a79e, 0x4f914f3c, 0x9f229e78, 0x3e453cf1,
   0x7c8a79e2, 0xf914f3c4, 0xf229e789, 0xe453cf13, 0xc8a79e27, 0x914f3c4f, 0x229e789f, 0x453cf13e,
   0x8a79e27c, 0x14f3c4f9, 0x29e789f2, 0x53cf13e4, 0xa79e27c8, 0x4f3c4f91, 0x9e789f22, 0x3cf13e45,
   0x79e27c8a, 0xf3c4f914, 0xe789f229, 0xcf13e453 },
   { 0x78df30ec, 0xf1be61d8, 0xe37cc3b1, 0xc6f98763, 0x8df30ec7, 0x1be61d8f, 0x37cc3b1e, 0x6f98763c,
   0xdf30ec78, 0xbe61d8f1, 0x7cc3b1e3, 0xf98763c6, 0xf30ec78d, 0xe61d8f1b, 0xcc3b1e37, 0x98763c6f,
   0x30ec78df, 0x61d8f1be, 0xc3b1e37c, 0x8763c6f9, 0x0ec78df3, 0x1d8f1be6, 0x3b1e37cc, 0x763c6f98,
   0xec78df30, 0xd8f1be61, 0xb1e37cc3, 0x63c6f987, 0xc78df30e, 0x8f1be61d, 0x1e37cc3b, 0x3c6f9876,
   0x78df30ec, 0xf1be61d8, 0xe37cc3b1, 0xc6f98763 },
   { 0x715ea49e, 0xe2bd493c, 0xc57a9279, 0x8af524f3, 0x15ea49e7, 0x2bd493ce, 0x57a9279c, 0xaf524f38,
   0x5ea49e71, 0xbd493ce2, 0x7a9279c5, 0xf524f38a, 0xea49e715, 0xd493ce2b, 0xa9279c57, 0x524f38af,
   0xa49e715e, 0x493ce2bd, 0x9279c57a, 0x24f38af5, 0x49e715ea, 0x93ce2bd4, 0x279c57a9, 0x4f38af52,
   0x9e715ea4, 0x3ce2bd49, 0x79c57a92, 0xf38af524, 0xe715ea49, 0xce2bd493, 0x9c57a927, 0x38af524f,
   0x715ea49e, 0xe2bd493c, 0xc57a9279, 0x8af524f3 },
   { 0xc785da0a, 0x8f0bb415, 0x1e17682b, 0x3c2ed056, 0x785da0ac, 0xf0bb4158, 0xe17682b1, 0xc2ed0563,
   0x85da0ac7, 0x0bb4158f, 0x17682b1e, 0x2ed0563c, 0x5da0ac78, 0xbb4158f0, 0x7682b1e1, 0xed0563c2,
   0xda0ac785, 0xb4158f0b, 0x682b1e17, 0xd0563c2e, 0xa0ac785d, 0x4158f0bb, 0x82b1e176, 0x0563c2ed,
   0x0ac785da, 0x158f0bb4, 0x2b1e1768, 0x563c2ed0, 0xac785da0, 0x58f0bb41, 0xb1e17682, 0x63c2ed05,
   0xc785da0a, 0x8f0bb415, 0x1e17682b, 0x3c2ed056 },
   { 0xe04ef22a, 0xc09de455, 0x813bc8ab, 0x02779157, 0x04ef22ae, 0x09de455c, 0x13bc8ab8, 0x27791570,
   0x4ef22ae0, 0x9de455c0, 0x3bc8ab81, 0x77915702, 0xef22ae04, 0xde455c09, 0xbc8ab813, 0x79157027,
   0xf22ae04e, 0xe455c09d, 0xc8ab813b, 0x91570277, 0x22ae04ef, 0x455c09de, 0x8ab813bc, 0x15702779,
   0x2ae04ef2, 0x55c09de4, 0xab813bc8, 0x57027791, 0xae04ef22, 0x5c09de45, 0xb813bc8a, 0x70277915,
   0xe04ef22a, 0xc09de455, 0x813bc8ab, 0x02779157 },
   { 0xe5c40957, 0xcb8812af, 0x9710255f, 0x2e204abf, 0x5c40957e, 0xb8812afc, 0x710255f9, 0xe204abf2,
   0xc40957e5, 0x8812afcb, 0x10255f97, 0x204abf2e, 0x40957e5c, 0x812afcb8, 0x0255f971, 0x04abf2e2,
   0x0957e5c4, 0x12afcb88, 0x255f9710, 0x4abf2e20, 0x957e5c40, 0x2afcb881, 0x55f97102, 0xabf2e204,
   0x57e5c409, 0xafcb8812, 0x5f971025, 0xbf2e204a, 0x7e5c4095, 0xfcb8812a, 0xf9710255, 0xf2e204ab,
   0xe5c40957, 0xcb8812af, 0x9710255f, 0x2e204abf }
};
__host__ __device__  void lea_set_key(LEA_KEY* key, const uint8_t* mk, unsigned int mk_len)
{
    if (!key)
        return;
    else if (!mk)
        return;

    switch (mk_len)
    {
    case 16:
        key->rk[0] = ROL(*((unsigned int*)mk) + delta[0][0], 1);
        key->rk[6] = ROL(key->rk[0] + delta[1][1], 1);
        key->rk[12] = ROL(key->rk[6] + delta[2][2], 1);
        key->rk[18] = ROL(key->rk[12] + delta[3][3], 1);
        key->rk[24] = ROL(key->rk[18] + delta[0][4], 1);
        key->rk[30] = ROL(key->rk[24] + delta[1][5], 1);
        key->rk[36] = ROL(key->rk[30] + delta[2][6], 1);
        key->rk[42] = ROL(key->rk[36] + delta[3][7], 1);
        key->rk[48] = ROL(key->rk[42] + delta[0][8], 1);
        key->rk[54] = ROL(key->rk[48] + delta[1][9], 1);
        key->rk[60] = ROL(key->rk[54] + delta[2][10], 1);
        key->rk[66] = ROL(key->rk[60] + delta[3][11], 1);
        key->rk[72] = ROL(key->rk[66] + delta[0][12], 1);
        key->rk[78] = ROL(key->rk[72] + delta[1][13], 1);
        key->rk[84] = ROL(key->rk[78] + delta[2][14], 1);
        key->rk[90] = ROL(key->rk[84] + delta[3][15], 1);
        key->rk[96] = ROL(key->rk[90] + delta[0][16], 1);
        key->rk[102] = ROL(key->rk[96] + delta[1][17], 1);
        key->rk[108] = ROL(key->rk[102] + delta[2][18], 1);
        key->rk[114] = ROL(key->rk[108] + delta[3][19], 1);
        key->rk[120] = ROL(key->rk[114] + delta[0][20], 1);
        key->rk[126] = ROL(key->rk[120] + delta[1][21], 1);
        key->rk[132] = ROL(key->rk[126] + delta[2][22], 1);
        key->rk[138] = ROL(key->rk[132] + delta[3][23], 1);

        key->rk[1] = key->rk[3] = key->rk[5] = ROL(*((unsigned int*)mk + 1) + delta[0][1], 3);
        key->rk[7] = key->rk[9] = key->rk[11] = ROL(key->rk[1] + delta[1][2], 3);
        key->rk[13] = key->rk[15] = key->rk[17] = ROL(key->rk[7] + delta[2][3], 3);
        key->rk[19] = key->rk[21] = key->rk[23] = ROL(key->rk[13] + delta[3][4], 3);
        key->rk[25] = key->rk[27] = key->rk[29] = ROL(key->rk[19] + delta[0][5], 3);
        key->rk[31] = key->rk[33] = key->rk[35] = ROL(key->rk[25] + delta[1][6], 3);
        key->rk[37] = key->rk[39] = key->rk[41] = ROL(key->rk[31] + delta[2][7], 3);
        key->rk[43] = key->rk[45] = key->rk[47] = ROL(key->rk[37] + delta[3][8], 3);
        key->rk[49] = key->rk[51] = key->rk[53] = ROL(key->rk[43] + delta[0][9], 3);
        key->rk[55] = key->rk[57] = key->rk[59] = ROL(key->rk[49] + delta[1][10], 3);
        key->rk[61] = key->rk[63] = key->rk[65] = ROL(key->rk[55] + delta[2][11], 3);
        key->rk[67] = key->rk[69] = key->rk[71] = ROL(key->rk[61] + delta[3][12], 3);
        key->rk[73] = key->rk[75] = key->rk[77] = ROL(key->rk[67] + delta[0][13], 3);
        key->rk[79] = key->rk[81] = key->rk[83] = ROL(key->rk[73] + delta[1][14], 3);
        key->rk[85] = key->rk[87] = key->rk[89] = ROL(key->rk[79] + delta[2][15], 3);
        key->rk[91] = key->rk[93] = key->rk[95] = ROL(key->rk[85] + delta[3][16], 3);
        key->rk[97] = key->rk[99] = key->rk[101] = ROL(key->rk[91] + delta[0][17], 3);
        key->rk[103] = key->rk[105] = key->rk[107] = ROL(key->rk[97] + delta[1][18], 3);
        key->rk[109] = key->rk[111] = key->rk[113] = ROL(key->rk[103] + delta[2][19], 3);
        key->rk[115] = key->rk[117] = key->rk[119] = ROL(key->rk[109] + delta[3][20], 3);
        key->rk[121] = key->rk[123] = key->rk[125] = ROL(key->rk[115] + delta[0][21], 3);
        key->rk[127] = key->rk[129] = key->rk[131] = ROL(key->rk[121] + delta[1][22], 3);
        key->rk[133] = key->rk[135] = key->rk[137] = ROL(key->rk[127] + delta[2][23], 3);
        key->rk[139] = key->rk[141] = key->rk[143] = ROL(key->rk[133] + delta[3][24], 3);

        key->rk[2] = ROL(*((unsigned int*)mk + 2) + delta[0][2], 6);
        key->rk[8] = ROL(key->rk[2] + delta[1][3], 6);
        key->rk[14] = ROL(key->rk[8] + delta[2][4], 6);
        key->rk[20] = ROL(key->rk[14] + delta[3][5], 6);
        key->rk[26] = ROL(key->rk[20] + delta[0][6], 6);
        key->rk[32] = ROL(key->rk[26] + delta[1][7], 6);
        key->rk[38] = ROL(key->rk[32] + delta[2][8], 6);
        key->rk[44] = ROL(key->rk[38] + delta[3][9], 6);
        key->rk[50] = ROL(key->rk[44] + delta[0][10], 6);
        key->rk[56] = ROL(key->rk[50] + delta[1][11], 6);
        key->rk[62] = ROL(key->rk[56] + delta[2][12], 6);
        key->rk[68] = ROL(key->rk[62] + delta[3][13], 6);
        key->rk[74] = ROL(key->rk[68] + delta[0][14], 6);
        key->rk[80] = ROL(key->rk[74] + delta[1][15], 6);
        key->rk[86] = ROL(key->rk[80] + delta[2][16], 6);
        key->rk[92] = ROL(key->rk[86] + delta[3][17], 6);
        key->rk[98] = ROL(key->rk[92] + delta[0][18], 6);
        key->rk[104] = ROL(key->rk[98] + delta[1][19], 6);
        key->rk[110] = ROL(key->rk[104] + delta[2][20], 6);
        key->rk[116] = ROL(key->rk[110] + delta[3][21], 6);
        key->rk[122] = ROL(key->rk[116] + delta[0][22], 6);
        key->rk[128] = ROL(key->rk[122] + delta[1][23], 6);
        key->rk[134] = ROL(key->rk[128] + delta[2][24], 6);
        key->rk[140] = ROL(key->rk[134] + delta[3][25], 6);

        key->rk[4] = ROL(*((unsigned int*)mk + 3) + delta[0][3], 11);
        key->rk[10] = ROL(key->rk[4] + delta[1][4], 11);
        key->rk[16] = ROL(key->rk[10] + delta[2][5], 11);
        key->rk[22] = ROL(key->rk[16] + delta[3][6], 11);
        key->rk[28] = ROL(key->rk[22] + delta[0][7], 11);
        key->rk[34] = ROL(key->rk[28] + delta[1][8], 11);
        key->rk[40] = ROL(key->rk[34] + delta[2][9], 11);
        key->rk[46] = ROL(key->rk[40] + delta[3][10], 11);
        key->rk[52] = ROL(key->rk[46] + delta[0][11], 11);
        key->rk[58] = ROL(key->rk[52] + delta[1][12], 11);
        key->rk[64] = ROL(key->rk[58] + delta[2][13], 11);
        key->rk[70] = ROL(key->rk[64] + delta[3][14], 11);
        key->rk[76] = ROL(key->rk[70] + delta[0][15], 11);
        key->rk[82] = ROL(key->rk[76] + delta[1][16], 11);
        key->rk[88] = ROL(key->rk[82] + delta[2][17], 11);
        key->rk[94] = ROL(key->rk[88] + delta[3][18], 11);
        key->rk[100] = ROL(key->rk[94] + delta[0][19], 11);
        key->rk[106] = ROL(key->rk[100] + delta[1][20], 11);
        key->rk[112] = ROL(key->rk[106] + delta[2][21], 11);
        key->rk[118] = ROL(key->rk[112] + delta[3][22], 11);
        key->rk[124] = ROL(key->rk[118] + delta[0][23], 11);
        key->rk[130] = ROL(key->rk[124] + delta[1][24], 11);
        key->rk[136] = ROL(key->rk[130] + delta[2][25], 11);
        key->rk[142] = ROL(key->rk[136] + delta[3][26], 11);
        break;

    case 24:
        key->rk[0] = ROL(*((unsigned int*)mk) + delta[0][0], 1);
        key->rk[6] = ROL(key->rk[0] + delta[1][1], 1);
        key->rk[12] = ROL(key->rk[6] + delta[2][2], 1);
        key->rk[18] = ROL(key->rk[12] + delta[3][3], 1);
        key->rk[24] = ROL(key->rk[18] + delta[4][4], 1);
        key->rk[30] = ROL(key->rk[24] + delta[5][5], 1);
        key->rk[36] = ROL(key->rk[30] + delta[0][6], 1);
        key->rk[42] = ROL(key->rk[36] + delta[1][7], 1);
        key->rk[48] = ROL(key->rk[42] + delta[2][8], 1);
        key->rk[54] = ROL(key->rk[48] + delta[3][9], 1);
        key->rk[60] = ROL(key->rk[54] + delta[4][10], 1);
        key->rk[66] = ROL(key->rk[60] + delta[5][11], 1);
        key->rk[72] = ROL(key->rk[66] + delta[0][12], 1);
        key->rk[78] = ROL(key->rk[72] + delta[1][13], 1);
        key->rk[84] = ROL(key->rk[78] + delta[2][14], 1);
        key->rk[90] = ROL(key->rk[84] + delta[3][15], 1);
        key->rk[96] = ROL(key->rk[90] + delta[4][16], 1);
        key->rk[102] = ROL(key->rk[96] + delta[5][17], 1);
        key->rk[108] = ROL(key->rk[102] + delta[0][18], 1);
        key->rk[114] = ROL(key->rk[108] + delta[1][19], 1);
        key->rk[120] = ROL(key->rk[114] + delta[2][20], 1);
        key->rk[126] = ROL(key->rk[120] + delta[3][21], 1);
        key->rk[132] = ROL(key->rk[126] + delta[4][22], 1);
        key->rk[138] = ROL(key->rk[132] + delta[5][23], 1);
        key->rk[144] = ROL(key->rk[138] + delta[0][24], 1);
        key->rk[150] = ROL(key->rk[144] + delta[1][25], 1);
        key->rk[156] = ROL(key->rk[150] + delta[2][26], 1);
        key->rk[162] = ROL(key->rk[156] + delta[3][27], 1);

        key->rk[1] = ROL(*((unsigned int*)mk + 1) + delta[0][1], 3);
        key->rk[7] = ROL(key->rk[1] + delta[1][2], 3);
        key->rk[13] = ROL(key->rk[7] + delta[2][3], 3);
        key->rk[19] = ROL(key->rk[13] + delta[3][4], 3);
        key->rk[25] = ROL(key->rk[19] + delta[4][5], 3);
        key->rk[31] = ROL(key->rk[25] + delta[5][6], 3);
        key->rk[37] = ROL(key->rk[31] + delta[0][7], 3);
        key->rk[43] = ROL(key->rk[37] + delta[1][8], 3);
        key->rk[49] = ROL(key->rk[43] + delta[2][9], 3);
        key->rk[55] = ROL(key->rk[49] + delta[3][10], 3);
        key->rk[61] = ROL(key->rk[55] + delta[4][11], 3);
        key->rk[67] = ROL(key->rk[61] + delta[5][12], 3);
        key->rk[73] = ROL(key->rk[67] + delta[0][13], 3);
        key->rk[79] = ROL(key->rk[73] + delta[1][14], 3);
        key->rk[85] = ROL(key->rk[79] + delta[2][15], 3);
        key->rk[91] = ROL(key->rk[85] + delta[3][16], 3);
        key->rk[97] = ROL(key->rk[91] + delta[4][17], 3);
        key->rk[103] = ROL(key->rk[97] + delta[5][18], 3);
        key->rk[109] = ROL(key->rk[103] + delta[0][19], 3);
        key->rk[115] = ROL(key->rk[109] + delta[1][20], 3);
        key->rk[121] = ROL(key->rk[115] + delta[2][21], 3);
        key->rk[127] = ROL(key->rk[121] + delta[3][22], 3);
        key->rk[133] = ROL(key->rk[127] + delta[4][23], 3);
        key->rk[139] = ROL(key->rk[133] + delta[5][24], 3);
        key->rk[145] = ROL(key->rk[139] + delta[0][25], 3);
        key->rk[151] = ROL(key->rk[145] + delta[1][26], 3);
        key->rk[157] = ROL(key->rk[151] + delta[2][27], 3);
        key->rk[163] = ROL(key->rk[157] + delta[3][28], 3);

        key->rk[2] = ROL(*((unsigned int*)mk + 2) + delta[0][2], 6);
        key->rk[8] = ROL(key->rk[2] + delta[1][3], 6);
        key->rk[14] = ROL(key->rk[8] + delta[2][4], 6);
        key->rk[20] = ROL(key->rk[14] + delta[3][5], 6);
        key->rk[26] = ROL(key->rk[20] + delta[4][6], 6);
        key->rk[32] = ROL(key->rk[26] + delta[5][7], 6);
        key->rk[38] = ROL(key->rk[32] + delta[0][8], 6);
        key->rk[44] = ROL(key->rk[38] + delta[1][9], 6);
        key->rk[50] = ROL(key->rk[44] + delta[2][10], 6);
        key->rk[56] = ROL(key->rk[50] + delta[3][11], 6);
        key->rk[62] = ROL(key->rk[56] + delta[4][12], 6);
        key->rk[68] = ROL(key->rk[62] + delta[5][13], 6);
        key->rk[74] = ROL(key->rk[68] + delta[0][14], 6);
        key->rk[80] = ROL(key->rk[74] + delta[1][15], 6);
        key->rk[86] = ROL(key->rk[80] + delta[2][16], 6);
        key->rk[92] = ROL(key->rk[86] + delta[3][17], 6);
        key->rk[98] = ROL(key->rk[92] + delta[4][18], 6);
        key->rk[104] = ROL(key->rk[98] + delta[5][19], 6);
        key->rk[110] = ROL(key->rk[104] + delta[0][20], 6);
        key->rk[116] = ROL(key->rk[110] + delta[1][21], 6);
        key->rk[122] = ROL(key->rk[116] + delta[2][22], 6);
        key->rk[128] = ROL(key->rk[122] + delta[3][23], 6);
        key->rk[134] = ROL(key->rk[128] + delta[4][24], 6);
        key->rk[140] = ROL(key->rk[134] + delta[5][25], 6);
        key->rk[146] = ROL(key->rk[140] + delta[0][26], 6);
        key->rk[152] = ROL(key->rk[146] + delta[1][27], 6);
        key->rk[158] = ROL(key->rk[152] + delta[2][28], 6);
        key->rk[164] = ROL(key->rk[158] + delta[3][29], 6);

        key->rk[3] = ROL(*((unsigned int*)mk + 3) + delta[0][3], 11);
        key->rk[9] = ROL(key->rk[3] + delta[1][4], 11);
        key->rk[15] = ROL(key->rk[9] + delta[2][5], 11);
        key->rk[21] = ROL(key->rk[15] + delta[3][6], 11);
        key->rk[27] = ROL(key->rk[21] + delta[4][7], 11);
        key->rk[33] = ROL(key->rk[27] + delta[5][8], 11);
        key->rk[39] = ROL(key->rk[33] + delta[0][9], 11);
        key->rk[45] = ROL(key->rk[39] + delta[1][10], 11);
        key->rk[51] = ROL(key->rk[45] + delta[2][11], 11);
        key->rk[57] = ROL(key->rk[51] + delta[3][12], 11);
        key->rk[63] = ROL(key->rk[57] + delta[4][13], 11);
        key->rk[69] = ROL(key->rk[63] + delta[5][14], 11);
        key->rk[75] = ROL(key->rk[69] + delta[0][15], 11);
        key->rk[81] = ROL(key->rk[75] + delta[1][16], 11);
        key->rk[87] = ROL(key->rk[81] + delta[2][17], 11);
        key->rk[93] = ROL(key->rk[87] + delta[3][18], 11);
        key->rk[99] = ROL(key->rk[93] + delta[4][19], 11);
        key->rk[105] = ROL(key->rk[99] + delta[5][20], 11);
        key->rk[111] = ROL(key->rk[105] + delta[0][21], 11);
        key->rk[117] = ROL(key->rk[111] + delta[1][22], 11);
        key->rk[123] = ROL(key->rk[117] + delta[2][23], 11);
        key->rk[129] = ROL(key->rk[123] + delta[3][24], 11);
        key->rk[135] = ROL(key->rk[129] + delta[4][25], 11);
        key->rk[141] = ROL(key->rk[135] + delta[5][26], 11);
        key->rk[147] = ROL(key->rk[141] + delta[0][27], 11);
        key->rk[153] = ROL(key->rk[147] + delta[1][28], 11);
        key->rk[159] = ROL(key->rk[153] + delta[2][29], 11);
        key->rk[165] = ROL(key->rk[159] + delta[3][30], 11);

        key->rk[4] = ROL(*((unsigned int*)mk + 4) + delta[0][4], 13);
        key->rk[10] = ROL(key->rk[4] + delta[1][5], 13);
        key->rk[16] = ROL(key->rk[10] + delta[2][6], 13);
        key->rk[22] = ROL(key->rk[16] + delta[3][7], 13);
        key->rk[28] = ROL(key->rk[22] + delta[4][8], 13);
        key->rk[34] = ROL(key->rk[28] + delta[5][9], 13);
        key->rk[40] = ROL(key->rk[34] + delta[0][10], 13);
        key->rk[46] = ROL(key->rk[40] + delta[1][11], 13);
        key->rk[52] = ROL(key->rk[46] + delta[2][12], 13);
        key->rk[58] = ROL(key->rk[52] + delta[3][13], 13);
        key->rk[64] = ROL(key->rk[58] + delta[4][14], 13);
        key->rk[70] = ROL(key->rk[64] + delta[5][15], 13);
        key->rk[76] = ROL(key->rk[70] + delta[0][16], 13);
        key->rk[82] = ROL(key->rk[76] + delta[1][17], 13);
        key->rk[88] = ROL(key->rk[82] + delta[2][18], 13);
        key->rk[94] = ROL(key->rk[88] + delta[3][19], 13);
        key->rk[100] = ROL(key->rk[94] + delta[4][20], 13);
        key->rk[106] = ROL(key->rk[100] + delta[5][21], 13);
        key->rk[112] = ROL(key->rk[106] + delta[0][22], 13);
        key->rk[118] = ROL(key->rk[112] + delta[1][23], 13);
        key->rk[124] = ROL(key->rk[118] + delta[2][24], 13);
        key->rk[130] = ROL(key->rk[124] + delta[3][25], 13);
        key->rk[136] = ROL(key->rk[130] + delta[4][26], 13);
        key->rk[142] = ROL(key->rk[136] + delta[5][27], 13);
        key->rk[148] = ROL(key->rk[142] + delta[0][28], 13);
        key->rk[154] = ROL(key->rk[148] + delta[1][29], 13);
        key->rk[160] = ROL(key->rk[154] + delta[2][30], 13);
        key->rk[166] = ROL(key->rk[160] + delta[3][31], 13);

        key->rk[5] = ROL(*((unsigned int*)mk + 5) + delta[0][5], 17);
        key->rk[11] = ROL(key->rk[5] + delta[1][6], 17);
        key->rk[17] = ROL(key->rk[11] + delta[2][7], 17);
        key->rk[23] = ROL(key->rk[17] + delta[3][8], 17);
        key->rk[29] = ROL(key->rk[23] + delta[4][9], 17);
        key->rk[35] = ROL(key->rk[29] + delta[5][10], 17);
        key->rk[41] = ROL(key->rk[35] + delta[0][11], 17);
        key->rk[47] = ROL(key->rk[41] + delta[1][12], 17);
        key->rk[53] = ROL(key->rk[47] + delta[2][13], 17);
        key->rk[59] = ROL(key->rk[53] + delta[3][14], 17);
        key->rk[65] = ROL(key->rk[59] + delta[4][15], 17);
        key->rk[71] = ROL(key->rk[65] + delta[5][16], 17);
        key->rk[77] = ROL(key->rk[71] + delta[0][17], 17);
        key->rk[83] = ROL(key->rk[77] + delta[1][18], 17);
        key->rk[89] = ROL(key->rk[83] + delta[2][19], 17);
        key->rk[95] = ROL(key->rk[89] + delta[3][20], 17);
        key->rk[101] = ROL(key->rk[95] + delta[4][21], 17);
        key->rk[107] = ROL(key->rk[101] + delta[5][22], 17);
        key->rk[113] = ROL(key->rk[107] + delta[0][23], 17);
        key->rk[119] = ROL(key->rk[113] + delta[1][24], 17);
        key->rk[125] = ROL(key->rk[119] + delta[2][25], 17);
        key->rk[131] = ROL(key->rk[125] + delta[3][26], 17);
        key->rk[137] = ROL(key->rk[131] + delta[4][27], 17);
        key->rk[143] = ROL(key->rk[137] + delta[5][28], 17);
        key->rk[149] = ROL(key->rk[143] + delta[0][29], 17);
        key->rk[155] = ROL(key->rk[149] + delta[1][30], 17);
        key->rk[161] = ROL(key->rk[155] + delta[2][31], 17);
        key->rk[167] = ROL(key->rk[161] + delta[3][0], 17);
        break;

    case 32:
        key->rk[0] = ROL(*((unsigned int*)mk) + delta[0][0], 1);
        key->rk[8] = ROL(key->rk[0] + delta[1][3], 6);
        key->rk[16] = ROL(key->rk[8] + delta[2][6], 13);
        key->rk[24] = ROL(key->rk[16] + delta[4][4], 1);
        key->rk[32] = ROL(key->rk[24] + delta[5][7], 6);
        key->rk[40] = ROL(key->rk[32] + delta[6][10], 13);
        key->rk[48] = ROL(key->rk[40] + delta[0][8], 1);
        key->rk[56] = ROL(key->rk[48] + delta[1][11], 6);
        key->rk[64] = ROL(key->rk[56] + delta[2][14], 13);
        key->rk[72] = ROL(key->rk[64] + delta[4][12], 1);
        key->rk[80] = ROL(key->rk[72] + delta[5][15], 6);
        key->rk[88] = ROL(key->rk[80] + delta[6][18], 13);
        key->rk[96] = ROL(key->rk[88] + delta[0][16], 1);
        key->rk[104] = ROL(key->rk[96] + delta[1][19], 6);
        key->rk[112] = ROL(key->rk[104] + delta[2][22], 13);
        key->rk[120] = ROL(key->rk[112] + delta[4][20], 1);
        key->rk[128] = ROL(key->rk[120] + delta[5][23], 6);
        key->rk[136] = ROL(key->rk[128] + delta[6][26], 13);
        key->rk[144] = ROL(key->rk[136] + delta[0][24], 1);
        key->rk[152] = ROL(key->rk[144] + delta[1][27], 6);
        key->rk[160] = ROL(key->rk[152] + delta[2][30], 13);
        key->rk[168] = ROL(key->rk[160] + delta[4][28], 1);
        key->rk[176] = ROL(key->rk[168] + delta[5][31], 6);
        key->rk[184] = ROL(key->rk[176] + delta[6][2], 13);

        key->rk[1] = ROL(*((unsigned int*)mk + 1) + delta[0][1], 3);
        key->rk[9] = ROL(key->rk[1] + delta[1][4], 11);
        key->rk[17] = ROL(key->rk[9] + delta[2][7], 17);
        key->rk[25] = ROL(key->rk[17] + delta[4][5], 3);
        key->rk[33] = ROL(key->rk[25] + delta[5][8], 11);
        key->rk[41] = ROL(key->rk[33] + delta[6][11], 17);
        key->rk[49] = ROL(key->rk[41] + delta[0][9], 3);
        key->rk[57] = ROL(key->rk[49] + delta[1][12], 11);
        key->rk[65] = ROL(key->rk[57] + delta[2][15], 17);
        key->rk[73] = ROL(key->rk[65] + delta[4][13], 3);
        key->rk[81] = ROL(key->rk[73] + delta[5][16], 11);
        key->rk[89] = ROL(key->rk[81] + delta[6][19], 17);
        key->rk[97] = ROL(key->rk[89] + delta[0][17], 3);
        key->rk[105] = ROL(key->rk[97] + delta[1][20], 11);
        key->rk[113] = ROL(key->rk[105] + delta[2][23], 17);
        key->rk[121] = ROL(key->rk[113] + delta[4][21], 3);
        key->rk[129] = ROL(key->rk[121] + delta[5][24], 11);
        key->rk[137] = ROL(key->rk[129] + delta[6][27], 17);
        key->rk[145] = ROL(key->rk[137] + delta[0][25], 3);
        key->rk[153] = ROL(key->rk[145] + delta[1][28], 11);
        key->rk[161] = ROL(key->rk[153] + delta[2][31], 17);
        key->rk[169] = ROL(key->rk[161] + delta[4][29], 3);
        key->rk[177] = ROL(key->rk[169] + delta[5][0], 11);
        key->rk[185] = ROL(key->rk[177] + delta[6][3], 17);

        key->rk[2] = ROL(*((unsigned int*)mk + 2) + delta[0][2], 6);
        key->rk[10] = ROL(key->rk[2] + delta[1][5], 13);
        key->rk[18] = ROL(key->rk[10] + delta[3][3], 1);
        key->rk[26] = ROL(key->rk[18] + delta[4][6], 6);
        key->rk[34] = ROL(key->rk[26] + delta[5][9], 13);
        key->rk[42] = ROL(key->rk[34] + delta[7][7], 1);
        key->rk[50] = ROL(key->rk[42] + delta[0][10], 6);
        key->rk[58] = ROL(key->rk[50] + delta[1][13], 13);
        key->rk[66] = ROL(key->rk[58] + delta[3][11], 1);
        key->rk[74] = ROL(key->rk[66] + delta[4][14], 6);
        key->rk[82] = ROL(key->rk[74] + delta[5][17], 13);
        key->rk[90] = ROL(key->rk[82] + delta[7][15], 1);
        key->rk[98] = ROL(key->rk[90] + delta[0][18], 6);
        key->rk[106] = ROL(key->rk[98] + delta[1][21], 13);
        key->rk[114] = ROL(key->rk[106] + delta[3][19], 1);
        key->rk[122] = ROL(key->rk[114] + delta[4][22], 6);
        key->rk[130] = ROL(key->rk[122] + delta[5][25], 13);
        key->rk[138] = ROL(key->rk[130] + delta[7][23], 1);
        key->rk[146] = ROL(key->rk[138] + delta[0][26], 6);
        key->rk[154] = ROL(key->rk[146] + delta[1][29], 13);
        key->rk[162] = ROL(key->rk[154] + delta[3][27], 1);
        key->rk[170] = ROL(key->rk[162] + delta[4][30], 6);
        key->rk[178] = ROL(key->rk[170] + delta[5][1], 13);
        key->rk[186] = ROL(key->rk[178] + delta[7][31], 1);

        key->rk[3] = ROL(*((unsigned int*)mk + 3) + delta[0][3], 11);
        key->rk[11] = ROL(key->rk[3] + delta[1][6], 17);
        key->rk[19] = ROL(key->rk[11] + delta[3][4], 3);
        key->rk[27] = ROL(key->rk[19] + delta[4][7], 11);
        key->rk[35] = ROL(key->rk[27] + delta[5][10], 17);
        key->rk[43] = ROL(key->rk[35] + delta[7][8], 3);
        key->rk[51] = ROL(key->rk[43] + delta[0][11], 11);
        key->rk[59] = ROL(key->rk[51] + delta[1][14], 17);
        key->rk[67] = ROL(key->rk[59] + delta[3][12], 3);
        key->rk[75] = ROL(key->rk[67] + delta[4][15], 11);
        key->rk[83] = ROL(key->rk[75] + delta[5][18], 17);
        key->rk[91] = ROL(key->rk[83] + delta[7][16], 3);
        key->rk[99] = ROL(key->rk[91] + delta[0][19], 11);
        key->rk[107] = ROL(key->rk[99] + delta[1][22], 17);
        key->rk[115] = ROL(key->rk[107] + delta[3][20], 3);
        key->rk[123] = ROL(key->rk[115] + delta[4][23], 11);
        key->rk[131] = ROL(key->rk[123] + delta[5][26], 17);
        key->rk[139] = ROL(key->rk[131] + delta[7][24], 3);
        key->rk[147] = ROL(key->rk[139] + delta[0][27], 11);
        key->rk[155] = ROL(key->rk[147] + delta[1][30], 17);
        key->rk[163] = ROL(key->rk[155] + delta[3][28], 3);
        key->rk[171] = ROL(key->rk[163] + delta[4][31], 11);
        key->rk[179] = ROL(key->rk[171] + delta[5][2], 17);
        key->rk[187] = ROL(key->rk[179] + delta[7][0], 3);

        key->rk[4] = ROL(*((unsigned int*)mk + 4) + delta[0][4], 13);
        key->rk[12] = ROL(key->rk[4] + delta[2][2], 1);
        key->rk[20] = ROL(key->rk[12] + delta[3][5], 6);
        key->rk[28] = ROL(key->rk[20] + delta[4][8], 13);
        key->rk[36] = ROL(key->rk[28] + delta[6][6], 1);
        key->rk[44] = ROL(key->rk[36] + delta[7][9], 6);
        key->rk[52] = ROL(key->rk[44] + delta[0][12], 13);
        key->rk[60] = ROL(key->rk[52] + delta[2][10], 1);
        key->rk[68] = ROL(key->rk[60] + delta[3][13], 6);
        key->rk[76] = ROL(key->rk[68] + delta[4][16], 13);
        key->rk[84] = ROL(key->rk[76] + delta[6][14], 1);
        key->rk[92] = ROL(key->rk[84] + delta[7][17], 6);
        key->rk[100] = ROL(key->rk[92] + delta[0][20], 13);
        key->rk[108] = ROL(key->rk[100] + delta[2][18], 1);
        key->rk[116] = ROL(key->rk[108] + delta[3][21], 6);
        key->rk[124] = ROL(key->rk[116] + delta[4][24], 13);
        key->rk[132] = ROL(key->rk[124] + delta[6][22], 1);
        key->rk[140] = ROL(key->rk[132] + delta[7][25], 6);
        key->rk[148] = ROL(key->rk[140] + delta[0][28], 13);
        key->rk[156] = ROL(key->rk[148] + delta[2][26], 1);
        key->rk[164] = ROL(key->rk[156] + delta[3][29], 6);
        key->rk[172] = ROL(key->rk[164] + delta[4][0], 13);
        key->rk[180] = ROL(key->rk[172] + delta[6][30], 1);
        key->rk[188] = ROL(key->rk[180] + delta[7][1], 6);

        key->rk[5] = ROL(*((unsigned int*)mk + 5) + delta[0][5], 17);
        key->rk[13] = ROL(key->rk[5] + delta[2][3], 3);
        key->rk[21] = ROL(key->rk[13] + delta[3][6], 11);
        key->rk[29] = ROL(key->rk[21] + delta[4][9], 17);
        key->rk[37] = ROL(key->rk[29] + delta[6][7], 3);
        key->rk[45] = ROL(key->rk[37] + delta[7][10], 11);
        key->rk[53] = ROL(key->rk[45] + delta[0][13], 17);
        key->rk[61] = ROL(key->rk[53] + delta[2][11], 3);
        key->rk[69] = ROL(key->rk[61] + delta[3][14], 11);
        key->rk[77] = ROL(key->rk[69] + delta[4][17], 17);
        key->rk[85] = ROL(key->rk[77] + delta[6][15], 3);
        key->rk[93] = ROL(key->rk[85] + delta[7][18], 11);
        key->rk[101] = ROL(key->rk[93] + delta[0][21], 17);
        key->rk[109] = ROL(key->rk[101] + delta[2][19], 3);
        key->rk[117] = ROL(key->rk[109] + delta[3][22], 11);
        key->rk[125] = ROL(key->rk[117] + delta[4][25], 17);
        key->rk[133] = ROL(key->rk[125] + delta[6][23], 3);
        key->rk[141] = ROL(key->rk[133] + delta[7][26], 11);
        key->rk[149] = ROL(key->rk[141] + delta[0][29], 17);
        key->rk[157] = ROL(key->rk[149] + delta[2][27], 3);
        key->rk[165] = ROL(key->rk[157] + delta[3][30], 11);
        key->rk[173] = ROL(key->rk[165] + delta[4][1], 17);
        key->rk[181] = ROL(key->rk[173] + delta[6][31], 3);
        key->rk[189] = ROL(key->rk[181] + delta[7][2], 11);

        key->rk[6] = ROL(*((unsigned int*)mk + 6) + delta[1][1], 1);
        key->rk[14] = ROL(key->rk[6] + delta[2][4], 6);
        key->rk[22] = ROL(key->rk[14] + delta[3][7], 13);
        key->rk[30] = ROL(key->rk[22] + delta[5][5], 1);
        key->rk[38] = ROL(key->rk[30] + delta[6][8], 6);
        key->rk[46] = ROL(key->rk[38] + delta[7][11], 13);
        key->rk[54] = ROL(key->rk[46] + delta[1][9], 1);
        key->rk[62] = ROL(key->rk[54] + delta[2][12], 6);
        key->rk[70] = ROL(key->rk[62] + delta[3][15], 13);
        key->rk[78] = ROL(key->rk[70] + delta[5][13], 1);
        key->rk[86] = ROL(key->rk[78] + delta[6][16], 6);
        key->rk[94] = ROL(key->rk[86] + delta[7][19], 13);
        key->rk[102] = ROL(key->rk[94] + delta[1][17], 1);
        key->rk[110] = ROL(key->rk[102] + delta[2][20], 6);
        key->rk[118] = ROL(key->rk[110] + delta[3][23], 13);
        key->rk[126] = ROL(key->rk[118] + delta[5][21], 1);
        key->rk[134] = ROL(key->rk[126] + delta[6][24], 6);
        key->rk[142] = ROL(key->rk[134] + delta[7][27], 13);
        key->rk[150] = ROL(key->rk[142] + delta[1][25], 1);
        key->rk[158] = ROL(key->rk[150] + delta[2][28], 6);
        key->rk[166] = ROL(key->rk[158] + delta[3][31], 13);
        key->rk[174] = ROL(key->rk[166] + delta[5][29], 1);
        key->rk[182] = ROL(key->rk[174] + delta[6][0], 6);
        key->rk[190] = ROL(key->rk[182] + delta[7][3], 13);

        key->rk[7] = ROL(*((unsigned int*)mk + 7) + delta[1][2], 3);
        key->rk[15] = ROL(key->rk[7] + delta[2][5], 11);
        key->rk[23] = ROL(key->rk[15] + delta[3][8], 17);
        key->rk[31] = ROL(key->rk[23] + delta[5][6], 3);
        key->rk[39] = ROL(key->rk[31] + delta[6][9], 11);
        key->rk[47] = ROL(key->rk[39] + delta[7][12], 17);
        key->rk[55] = ROL(key->rk[47] + delta[1][10], 3);
        key->rk[63] = ROL(key->rk[55] + delta[2][13], 11);
        key->rk[71] = ROL(key->rk[63] + delta[3][16], 17);
        key->rk[79] = ROL(key->rk[71] + delta[5][14], 3);
        key->rk[87] = ROL(key->rk[79] + delta[6][17], 11);
        key->rk[95] = ROL(key->rk[87] + delta[7][20], 17);
        key->rk[103] = ROL(key->rk[95] + delta[1][18], 3);
        key->rk[111] = ROL(key->rk[103] + delta[2][21], 11);
        key->rk[119] = ROL(key->rk[111] + delta[3][24], 17);
        key->rk[127] = ROL(key->rk[119] + delta[5][22], 3);
        key->rk[135] = ROL(key->rk[127] + delta[6][25], 11);
        key->rk[143] = ROL(key->rk[135] + delta[7][28], 17);
        key->rk[151] = ROL(key->rk[143] + delta[1][26], 3);
        key->rk[159] = ROL(key->rk[151] + delta[2][29], 11);
        key->rk[167] = ROL(key->rk[159] + delta[3][0], 17);
        key->rk[175] = ROL(key->rk[167] + delta[5][30], 3);
        key->rk[183] = ROL(key->rk[175] + delta[6][1], 11);
        key->rk[191] = ROL(key->rk[183] + delta[7][4], 17);

        break;

    default:
        return;
    }

    key->round = (mk_len >> 1) + 16;
}

#define XOR8x16(r, a, b)            \
   *((r)      ) = *((a)      ) ^ *((b)      ),   \
   *((r) + 0x1) = *((a) + 0x1) ^ *((b) + 0x1),   \
   *((r) + 0x2) = *((a) + 0x2) ^ *((b) + 0x2),   \
   *((r) + 0x3) = *((a) + 0x3) ^ *((b) + 0x3),   \
   *((r) + 0x4) = *((a) + 0x4) ^ *((b) + 0x4),   \
   *((r) + 0x5) = *((a) + 0x5) ^ *((b) + 0x5),   \
   *((r) + 0x6) = *((a) + 0x6) ^ *((b) + 0x6),   \
   *((r) + 0x7) = *((a) + 0x7) ^ *((b) + 0x7),   \
   *((r) + 0x8) = *((a) + 0x8) ^ *((b) + 0x8),   \
   *((r) + 0x9) = *((a) + 0x9) ^ *((b) + 0x9),   \
   *((r) + 0xa) = *((a) + 0xa) ^ *((b) + 0xa),   \
   *((r) + 0xb) = *((a) + 0xb) ^ *((b) + 0xb),   \
   *((r) + 0xc) = *((a) + 0xc) ^ *((b) + 0xc),   \
   *((r) + 0xd) = *((a) + 0xd) ^ *((b) + 0xd),   \
   *((r) + 0xe) = *((a) + 0xe) ^ *((b) + 0xe),   \
   *((r) + 0xf) = *((a) + 0xf) ^ *((b) + 0xf)

#define RSHIFT8x16_1(v)                        \
   (v)[15] = ((v)[15] >> 1) | ((v)[14] << 7),      \
   (v)[14] = ((v)[14] >> 1) | ((v)[13] << 7),      \
   (v)[13] = ((v)[13] >> 1) | ((v)[12] << 7),      \
   (v)[12] = ((v)[12] >> 1) | ((v)[11] << 7),      \
   (v)[11] = ((v)[11] >> 1) | ((v)[10] << 7),      \
   (v)[10] = ((v)[10] >> 1) | ((v)[ 9] << 7),      \
   (v)[ 9] = ((v)[ 9] >> 1) | ((v)[ 8] << 7),      \
   (v)[ 8] = ((v)[ 8] >> 1) | ((v)[ 7] << 7),      \
   (v)[ 7] = ((v)[ 7] >> 1) | ((v)[ 6] << 7),      \
   (v)[ 6] = ((v)[ 6] >> 1) | ((v)[ 5] << 7),      \
   (v)[ 5] = ((v)[ 5] >> 1) | ((v)[ 4] << 7),      \
   (v)[ 4] = ((v)[ 4] >> 1) | ((v)[ 3] << 7),      \
   (v)[ 3] = ((v)[ 3] >> 1) | ((v)[ 2] << 7),      \
   (v)[ 2] = ((v)[ 2] >> 1) | ((v)[ 1] << 7),      \
   (v)[ 1] = ((v)[ 1] >> 1) | ((v)[ 0] << 7),      \
   (v)[ 0] = ((v)[ 0] >> 1)

#define RSHIFT8x16_4(v)                        \
   (v)[15] = ((v)[15] >> 4) | ((v)[14] << 4),      \
   (v)[14] = ((v)[14] >> 4) | ((v)[13] << 4),      \
   (v)[13] = ((v)[13] >> 4) | ((v)[12] << 4),      \
   (v)[12] = ((v)[12] >> 4) | ((v)[11] << 4),      \
   (v)[11] = ((v)[11] >> 4) | ((v)[10] << 4),      \
   (v)[10] = ((v)[10] >> 4) | ((v)[ 9] << 4),      \
   (v)[ 9] = ((v)[ 9] >> 4) | ((v)[ 8] << 4),      \
   (v)[ 8] = ((v)[ 8] >> 4) | ((v)[ 7] << 4),      \
   (v)[ 7] = ((v)[ 7] >> 4) | ((v)[ 6] << 4),      \
   (v)[ 6] = ((v)[ 6] >> 4) | ((v)[ 5] << 4),      \
   (v)[ 5] = ((v)[ 5] >> 4) | ((v)[ 4] << 4),      \
   (v)[ 4] = ((v)[ 4] >> 4) | ((v)[ 3] << 4),      \
   (v)[ 3] = ((v)[ 3] >> 4) | ((v)[ 2] << 4),      \
   (v)[ 2] = ((v)[ 2] >> 4) | ((v)[ 1] << 4),      \
   (v)[ 1] = ((v)[ 1] >> 4) | ((v)[ 0] << 4),      \
   (v)[ 0] = ((v)[ 0] >> 4)

#define RSHIFT8x16_8(v)                     \
   (v)[15] = (v)[14],      \
   (v)[14] = (v)[13],      \
   (v)[13] = (v)[12],      \
   (v)[12] = (v)[11],      \
   (v)[11] = (v)[10],      \
   (v)[10] = (v)[ 9],      \
   (v)[ 9] = (v)[ 8],      \
   (v)[ 8] = (v)[ 7],      \
   (v)[ 7] = (v)[ 6],      \
   (v)[ 6] = (v)[ 5],      \
   (v)[ 5] = (v)[ 4],      \
   (v)[ 4] = (v)[ 3],      \
   (v)[ 3] = (v)[ 2],      \
   (v)[ 2] = (v)[ 1],      \
   (v)[ 1] = (v)[ 0],      \
   (v)[ 0] = 0

#define CPY8x16(d, s)                                 \
   *((unsigned int *)(d)) = *((unsigned int *)(s)),         \
   *((unsigned int *)(d) + 1) = *((unsigned int *)(s) + 1),   \
   *((unsigned int *)(d) + 2) = *((unsigned int *)(s) + 2),   \
   *((unsigned int *)(d) + 3) = *((unsigned int *)(s) + 3)

__host__ __device__  void lea_encrypt(uint8_t* ct, const uint8_t* pt, const LEA_KEY* key)
{
    unsigned int X0, X1, X2, X3;

    const unsigned int* _pt = (const unsigned int*)pt;
    unsigned int* _ct = (unsigned int*)ct;


    X0 = *(unsigned int*)(_pt + 0);
    X1 = *(unsigned int*)(_pt + 1);
    X2 = *(unsigned int*)(_pt + 2);
    X3 = *(unsigned int*)(_pt + 3);



    X3 = ROR((X2 ^ key->rk[4]) + (X3 ^ key->rk[5]), 3);
    X2 = ROR((X1 ^ key->rk[2]) + (X2 ^ key->rk[3]), 5);
    X1 = ROL((X0 ^ key->rk[0]) + (X1 ^ key->rk[1]), 9);
    X0 = ROR((X3 ^ key->rk[10]) + (X0 ^ key->rk[11]), 3);
    X3 = ROR((X2 ^ key->rk[8]) + (X3 ^ key->rk[9]), 5);
    X2 = ROL((X1 ^ key->rk[6]) + (X2 ^ key->rk[7]), 9);
    X1 = ROR((X0 ^ key->rk[16]) + (X1 ^ key->rk[17]), 3);
    X0 = ROR((X3 ^ key->rk[14]) + (X0 ^ key->rk[15]), 5);
    X3 = ROL((X2 ^ key->rk[12]) + (X3 ^ key->rk[13]), 9);
    X2 = ROR((X1 ^ key->rk[22]) + (X2 ^ key->rk[23]), 3);
    X1 = ROR((X0 ^ key->rk[20]) + (X1 ^ key->rk[21]), 5);
    X0 = ROL((X3 ^ key->rk[18]) + (X0 ^ key->rk[19]), 9);

    X3 = ROR((X2 ^ key->rk[28]) + (X3 ^ key->rk[29]), 3);
    X2 = ROR((X1 ^ key->rk[26]) + (X2 ^ key->rk[27]), 5);
    X1 = ROL((X0 ^ key->rk[24]) + (X1 ^ key->rk[25]), 9);
    X0 = ROR((X3 ^ key->rk[34]) + (X0 ^ key->rk[35]), 3);
    X3 = ROR((X2 ^ key->rk[32]) + (X3 ^ key->rk[33]), 5);
    X2 = ROL((X1 ^ key->rk[30]) + (X2 ^ key->rk[31]), 9);
    X1 = ROR((X0 ^ key->rk[40]) + (X1 ^ key->rk[41]), 3);
    X0 = ROR((X3 ^ key->rk[38]) + (X0 ^ key->rk[39]), 5);
    X3 = ROL((X2 ^ key->rk[36]) + (X3 ^ key->rk[37]), 9);
    X2 = ROR((X1 ^ key->rk[46]) + (X2 ^ key->rk[47]), 3);
    X1 = ROR((X0 ^ key->rk[44]) + (X1 ^ key->rk[45]), 5);
    X0 = ROL((X3 ^ key->rk[42]) + (X0 ^ key->rk[43]), 9);

    X3 = ROR((X2 ^ key->rk[52]) + (X3 ^ key->rk[53]), 3);
    X2 = ROR((X1 ^ key->rk[50]) + (X2 ^ key->rk[51]), 5);
    X1 = ROL((X0 ^ key->rk[48]) + (X1 ^ key->rk[49]), 9);
    X0 = ROR((X3 ^ key->rk[58]) + (X0 ^ key->rk[59]), 3);
    X3 = ROR((X2 ^ key->rk[56]) + (X3 ^ key->rk[57]), 5);
    X2 = ROL((X1 ^ key->rk[54]) + (X2 ^ key->rk[55]), 9);
    X1 = ROR((X0 ^ key->rk[64]) + (X1 ^ key->rk[65]), 3);
    X0 = ROR((X3 ^ key->rk[62]) + (X0 ^ key->rk[63]), 5);
    X3 = ROL((X2 ^ key->rk[60]) + (X3 ^ key->rk[61]), 9);
    X2 = ROR((X1 ^ key->rk[70]) + (X2 ^ key->rk[71]), 3);
    X1 = ROR((X0 ^ key->rk[68]) + (X1 ^ key->rk[69]), 5);
    X0 = ROL((X3 ^ key->rk[66]) + (X0 ^ key->rk[67]), 9);

    X3 = ROR((X2 ^ key->rk[76]) + (X3 ^ key->rk[77]), 3);
    X2 = ROR((X1 ^ key->rk[74]) + (X2 ^ key->rk[75]), 5);
    X1 = ROL((X0 ^ key->rk[72]) + (X1 ^ key->rk[73]), 9);
    X0 = ROR((X3 ^ key->rk[82]) + (X0 ^ key->rk[83]), 3);
    X3 = ROR((X2 ^ key->rk[80]) + (X3 ^ key->rk[81]), 5);
    X2 = ROL((X1 ^ key->rk[78]) + (X2 ^ key->rk[79]), 9);
    X1 = ROR((X0 ^ key->rk[88]) + (X1 ^ key->rk[89]), 3);
    X0 = ROR((X3 ^ key->rk[86]) + (X0 ^ key->rk[87]), 5);
    X3 = ROL((X2 ^ key->rk[84]) + (X3 ^ key->rk[85]), 9);
    X2 = ROR((X1 ^ key->rk[94]) + (X2 ^ key->rk[95]), 3);
    X1 = ROR((X0 ^ key->rk[92]) + (X1 ^ key->rk[93]), 5);
    X0 = ROL((X3 ^ key->rk[90]) + (X0 ^ key->rk[91]), 9);

    X3 = ROR((X2 ^ key->rk[100]) + (X3 ^ key->rk[101]), 3);
    X2 = ROR((X1 ^ key->rk[98]) + (X2 ^ key->rk[99]), 5);
    X1 = ROL((X0 ^ key->rk[96]) + (X1 ^ key->rk[97]), 9);
    X0 = ROR((X3 ^ key->rk[106]) + (X0 ^ key->rk[107]), 3);
    X3 = ROR((X2 ^ key->rk[104]) + (X3 ^ key->rk[105]), 5);
    X2 = ROL((X1 ^ key->rk[102]) + (X2 ^ key->rk[103]), 9);
    X1 = ROR((X0 ^ key->rk[112]) + (X1 ^ key->rk[113]), 3);
    X0 = ROR((X3 ^ key->rk[110]) + (X0 ^ key->rk[111]), 5);
    X3 = ROL((X2 ^ key->rk[108]) + (X3 ^ key->rk[109]), 9);
    X2 = ROR((X1 ^ key->rk[118]) + (X2 ^ key->rk[119]), 3);
    X1 = ROR((X0 ^ key->rk[116]) + (X1 ^ key->rk[117]), 5);
    X0 = ROL((X3 ^ key->rk[114]) + (X0 ^ key->rk[115]), 9);

    X3 = ROR((X2 ^ key->rk[124]) + (X3 ^ key->rk[125]), 3);
    X2 = ROR((X1 ^ key->rk[122]) + (X2 ^ key->rk[123]), 5);
    X1 = ROL((X0 ^ key->rk[120]) + (X1 ^ key->rk[121]), 9);
    X0 = ROR((X3 ^ key->rk[130]) + (X0 ^ key->rk[131]), 3);
    X3 = ROR((X2 ^ key->rk[128]) + (X3 ^ key->rk[129]), 5);
    X2 = ROL((X1 ^ key->rk[126]) + (X2 ^ key->rk[127]), 9);
    X1 = ROR((X0 ^ key->rk[136]) + (X1 ^ key->rk[137]), 3);
    X0 = ROR((X3 ^ key->rk[134]) + (X0 ^ key->rk[135]), 5);
    X3 = ROL((X2 ^ key->rk[132]) + (X3 ^ key->rk[133]), 9);
    X2 = ROR((X1 ^ key->rk[142]) + (X2 ^ key->rk[143]), 3);
    X1 = ROR((X0 ^ key->rk[140]) + (X1 ^ key->rk[141]), 5);
    X0 = ROL((X3 ^ key->rk[138]) + (X0 ^ key->rk[139]), 9);

    if (key->round > 24)
    {
        X3 = ROR((X2 ^ key->rk[148]) + (X3 ^ key->rk[149]), 3);
        X2 = ROR((X1 ^ key->rk[146]) + (X2 ^ key->rk[147]), 5);
        X1 = ROL((X0 ^ key->rk[144]) + (X1 ^ key->rk[145]), 9);
        X0 = ROR((X3 ^ key->rk[154]) + (X0 ^ key->rk[155]), 3);
        X3 = ROR((X2 ^ key->rk[152]) + (X3 ^ key->rk[153]), 5);
        X2 = ROL((X1 ^ key->rk[150]) + (X2 ^ key->rk[151]), 9);
        X1 = ROR((X0 ^ key->rk[160]) + (X1 ^ key->rk[161]), 3);
        X0 = ROR((X3 ^ key->rk[158]) + (X0 ^ key->rk[159]), 5);
        X3 = ROL((X2 ^ key->rk[156]) + (X3 ^ key->rk[157]), 9);
        X2 = ROR((X1 ^ key->rk[166]) + (X2 ^ key->rk[167]), 3);
        X1 = ROR((X0 ^ key->rk[164]) + (X1 ^ key->rk[165]), 5);
        X0 = ROL((X3 ^ key->rk[162]) + (X0 ^ key->rk[163]), 9);
    }
    if (key->round > 28)
    {
        X3 = ROR((X2 ^ key->rk[172]) + (X3 ^ key->rk[173]), 3);
        X2 = ROR((X1 ^ key->rk[170]) + (X2 ^ key->rk[171]), 5);
        X1 = ROL((X0 ^ key->rk[168]) + (X1 ^ key->rk[169]), 9);
        X0 = ROR((X3 ^ key->rk[178]) + (X0 ^ key->rk[179]), 3);
        X3 = ROR((X2 ^ key->rk[176]) + (X3 ^ key->rk[177]), 5);
        X2 = ROL((X1 ^ key->rk[174]) + (X2 ^ key->rk[175]), 9);
        X1 = ROR((X0 ^ key->rk[184]) + (X1 ^ key->rk[185]), 3);
        X0 = ROR((X3 ^ key->rk[182]) + (X0 ^ key->rk[183]), 5);
        X3 = ROL((X2 ^ key->rk[180]) + (X3 ^ key->rk[181]), 9);
        X2 = ROR((X1 ^ key->rk[190]) + (X2 ^ key->rk[191]), 3);
        X1 = ROR((X0 ^ key->rk[188]) + (X1 ^ key->rk[189]), 5);
        X0 = ROL((X3 ^ key->rk[186]) + (X0 ^ key->rk[187]), 9);
    }
    *(unsigned int*)(_ct + 0) = X0;
    *(unsigned int*)(_ct + 1) = X1;
    *(unsigned int*)(_ct + 2) = X2;
    *(unsigned int*)(_ct + 3) = X3;
}
//진짜 곱셈
__host__ __device__  static void gcm_gfmul_m(uint8_t* r, const uint8_t* x, const uint8_t* y)
{
    uint8_t z[16] = { 0x00, };
    uint8_t v[16] = { 0x00, };
    int i = 0;
    memcpy(v, y, 16);

    for (i = 0; i < 128; i++)
    {
        if ((x[i >> 3] >> (7 - (i & 0x7))) & 1) {
            //printf("%d , x[i >> 3] >> (7 - (i & 0x7))) & 1 : %d\n", i, ((x[i >> 3] >> (7 - (i & 0x7))) & 1));
            XOR8x16(z, z, v);
        }
        if (v[15] & 1)
        {
            RSHIFT8x16_1(v);

            v[0] ^= 0xe1;
        }
        else
            RSHIFT8x16_1(v);

    }

    memcpy(r, z, 16);
}
__host__ __device__  static void _gcm_ghash_m(uint8_t* r, const uint8_t* x, int x_len, const uint8_t h[][16])
{
    int i;
    uint8_t y[16] = { 0, };

    memcpy(y, r, 16);

    for (; x_len >= 16; x_len -= 16, x += 16)
    {
        XOR8x16(y, y, x);

        gcm_gfmul_m(y, y, h[0]);
    }

    if (x_len)
    {
        for (i = 0; i < x_len; i++)
            y[i] = y[i] ^ x[i];

        gcm_gfmul_m(y, y, h[0]);
    }

    memcpy(r, y, 16);
}
__host__ __device__  static void _lea_gcm_init_m(LEA_GCM_CTX* ctx, const uint8_t* mk, int mk_len)//ctx->h[0] = H (GHASH에서 쓰는 값)
{
    uint8_t zero[16] = { 0, };

    memset(ctx, 0, sizeof(LEA_GCM_CTX));

    lea_set_key(&ctx->key, mk, mk_len);
    lea_encrypt((uint8_t*)ctx->h, zero, &ctx->key);
}
//4bit table
__constant__ const unsigned char reduction_4bit[16][2] = {
    { 0x00, 0x00 }, { 0x1c, 0x20 }, { 0x38, 0x40 }, { 0x24, 0x60 }, { 0x70, 0x80 }, { 0x6c, 0xa0 }, { 0x48, 0xc0 }, { 0x54, 0xe0 },
    { 0xe1, 0x00 }, { 0xfd, 0x20 }, { 0xd9, 0x40 }, { 0xc5, 0x60 }, { 0x91, 0x80 }, { 0x8d, 0xa0 }, { 0xa9, 0xc0 }, { 0xb5, 0xe0 }
};
__host__ __device__  static void gcm_init_4bit_table(unsigned char hTable[][16], const unsigned char* h)
{
    unsigned char tmp[16];

    memcpy(tmp, h, 16);
    ;
    memcpy(hTable[8], tmp, 16);

    RSHIFT8x16_1(tmp);
    if (hTable[8][15] & 1)
        tmp[0] ^= 0xe1;
    memcpy(hTable[4], tmp, 16);

    RSHIFT8x16_1(tmp);
    if (hTable[4][15] & 1)
        tmp[0] ^= 0xe1;
    memcpy(hTable[2], tmp, 16);


    RSHIFT8x16_1(tmp);
    if (hTable[2][15] & 1)
        tmp[0] ^= 0xe1;
    memcpy(hTable[1], tmp, 16);


    XOR8x16(hTable[3], hTable[2], hTable[1]);
    XOR8x16(hTable[5], hTable[4], hTable[1]);
    XOR8x16(hTable[6], hTable[4], hTable[2]);
    XOR8x16(hTable[7], hTable[4], hTable[3]);
    XOR8x16(hTable[9], hTable[8], hTable[1]);
    XOR8x16(hTable[10], hTable[8], hTable[2]);
    XOR8x16(hTable[11], hTable[8], hTable[3]);
    XOR8x16(hTable[12], hTable[8], hTable[4]);
    XOR8x16(hTable[13], hTable[8], hTable[5]);
    XOR8x16(hTable[14], hTable[8], hTable[6]);
    XOR8x16(hTable[15], hTable[8], hTable[7]);
}
__host__ __device__  static void gcm_gfmul_4(unsigned char* r, const unsigned char* x, const unsigned char hTable[][16])
{
    unsigned char z[16], mask;
    int i;

    memset(z, 0, 16);

    for (i = 15; i > 0; i--)
    {
        mask = x[i] & 0xf;
        XOR8x16(z, z, hTable[mask]);

        mask = z[15] & 0xf;
        RSHIFT8x16_4(z);
        z[0] ^= reduction_4bit[mask][0];
        z[1] ^= reduction_4bit[mask][1];
        ////======
        mask = x[i] >> 4;
        XOR8x16(z, z, hTable[mask]);

        mask = z[15] & 0xf;
        RSHIFT8x16_4(z);
        z[0] ^= reduction_4bit[mask][0];
        z[1] ^= reduction_4bit[mask][1];

    }

    mask = x[i] & 0xf;
    XOR8x16(z, z, hTable[mask]);

    mask = z[15] & 0xf;
    RSHIFT8x16_4(z);
    z[0] ^= reduction_4bit[mask][0];
    z[1] ^= reduction_4bit[mask][1];

    mask = x[i] >> 4;
    XOR8x16(z, z, hTable[mask]);

    memcpy(r, z, 16);
}
__host__ __device__  static void _gcm_ghash_4(unsigned char* r, const unsigned char* x, unsigned int x_len, const unsigned char hTable[][16])
{
    unsigned int i;
    unsigned char y[16] = { 0, };

    memcpy(y, r, 16);

    for (; x_len >= 16; x_len -= 16, x += 16)
    {
        XOR8x16(y, y, x);

        gcm_gfmul_4(y, y, hTable);
    }

    if (x_len)
    {
        for (i = 0; i < x_len; i++)
            y[i] = y[i] ^ x[i];

        gcm_gfmul_4(y, y, hTable);
    }

    memcpy(r, y, 16);
}
__host__ __device__  static void _lea_gcm_init_4(LEA_GCM_CTX* ctx, const unsigned char* mk, int mk_len)
{
    unsigned char zero[16] = { 0, }, h[16] = { 0x00, };

    memset(ctx, 0, sizeof(LEA_GCM_CTX));

    lea_set_key(&ctx->key, mk, mk_len);
    lea_encrypt(zero, zero, &ctx->key);

    gcm_init_4bit_table(ctx->h, zero);
}
// 8bit 테이블 버전
__constant__ const unsigned char reduction_8bit[256][2] = {
    { 0x00, 0x00 }, { 0x01, 0xc2 }, { 0x03, 0x84 }, { 0x02, 0x46 }, { 0x07, 0x08 }, { 0x06, 0xca }, { 0x04, 0x8c }, { 0x05, 0x4e },
    { 0x0e, 0x10 }, { 0x0f, 0xd2 }, { 0x0d, 0x94 }, { 0x0c, 0x56 }, { 0x09, 0x18 }, { 0x08, 0xda }, { 0x0a, 0x9c }, { 0x0b, 0x5e },
    { 0x1c, 0x20 }, { 0x1d, 0xe2 }, { 0x1f, 0xa4 }, { 0x1e, 0x66 }, { 0x1b, 0x28 }, { 0x1a, 0xea }, { 0x18, 0xac }, { 0x19, 0x6e },
    { 0x12, 0x30 }, { 0x13, 0xf2 }, { 0x11, 0xb4 }, { 0x10, 0x76 }, { 0x15, 0x38 }, { 0x14, 0xfa }, { 0x16, 0xbc }, { 0x17, 0x7e },
    { 0x38, 0x40 }, { 0x39, 0x82 }, { 0x3b, 0xc4 }, { 0x3a, 0x06 }, { 0x3f, 0x48 }, { 0x3e, 0x8a }, { 0x3c, 0xcc }, { 0x3d, 0x0e },
    { 0x36, 0x50 }, { 0x37, 0x92 }, { 0x35, 0xd4 }, { 0x34, 0x16 }, { 0x31, 0x58 }, { 0x30, 0x9a }, { 0x32, 0xdc }, { 0x33, 0x1e },
    { 0x24, 0x60 }, { 0x25, 0xa2 }, { 0x27, 0xe4 }, { 0x26, 0x26 }, { 0x23, 0x68 }, { 0x22, 0xaa }, { 0x20, 0xec }, { 0x21, 0x2e },
    { 0x2a, 0x70 }, { 0x2b, 0xb2 }, { 0x29, 0xf4 }, { 0x28, 0x36 }, { 0x2d, 0x78 }, { 0x2c, 0xba }, { 0x2e, 0xfc }, { 0x2f, 0x3e },
    { 0x70, 0x80 }, { 0x71, 0x42 }, { 0x73, 0x04 }, { 0x72, 0xc6 }, { 0x77, 0x88 }, { 0x76, 0x4a }, { 0x74, 0x0c }, { 0x75, 0xce },
    { 0x7e, 0x90 }, { 0x7f, 0x52 }, { 0x7d, 0x14 }, { 0x7c, 0xd6 }, { 0x79, 0x98 }, { 0x78, 0x5a }, { 0x7a, 0x1c }, { 0x7b, 0xde },
    { 0x6c, 0xa0 }, { 0x6d, 0x62 }, { 0x6f, 0x24 }, { 0x6e, 0xe6 }, { 0x6b, 0xa8 }, { 0x6a, 0x6a }, { 0x68, 0x2c }, { 0x69, 0xee },
    { 0x62, 0xb0 }, { 0x63, 0x72 }, { 0x61, 0x34 }, { 0x60, 0xf6 }, { 0x65, 0xb8 }, { 0x64, 0x7a }, { 0x66, 0x3c }, { 0x67, 0xfe },
    { 0x48, 0xc0 }, { 0x49, 0x02 }, { 0x4b, 0x44 }, { 0x4a, 0x86 }, { 0x4f, 0xc8 }, { 0x4e, 0x0a }, { 0x4c, 0x4c }, { 0x4d, 0x8e },
    { 0x46, 0xd0 }, { 0x47, 0x12 }, { 0x45, 0x54 }, { 0x44, 0x96 }, { 0x41, 0xd8 }, { 0x40, 0x1a }, { 0x42, 0x5c }, { 0x43, 0x9e },
    { 0x54, 0xe0 }, { 0x55, 0x22 }, { 0x57, 0x64 }, { 0x56, 0xa6 }, { 0x53, 0xe8 }, { 0x52, 0x2a }, { 0x50, 0x6c }, { 0x51, 0xae },
    { 0x5a, 0xf0 }, { 0x5b, 0x32 }, { 0x59, 0x74 }, { 0x58, 0xb6 }, { 0x5d, 0xf8 }, { 0x5c, 0x3a }, { 0x5e, 0x7c }, { 0x5f, 0xbe },
    { 0xe1, 0x00 }, { 0xe0, 0xc2 }, { 0xe2, 0x84 }, { 0xe3, 0x46 }, { 0xe6, 0x08 }, { 0xe7, 0xca }, { 0xe5, 0x8c }, { 0xe4, 0x4e },
    { 0xef, 0x10 }, { 0xee, 0xd2 }, { 0xec, 0x94 }, { 0xed, 0x56 }, { 0xe8, 0x18 }, { 0xe9, 0xda }, { 0xeb, 0x9c }, { 0xea, 0x5e },
    { 0xfd, 0x20 }, { 0xfc, 0xe2 }, { 0xfe, 0xa4 }, { 0xff, 0x66 }, { 0xfa, 0x28 }, { 0xfb, 0xea }, { 0xf9, 0xac }, { 0xf8, 0x6e },
    { 0xf3, 0x30 }, { 0xf2, 0xf2 }, { 0xf0, 0xb4 }, { 0xf1, 0x76 }, { 0xf4, 0x38 }, { 0xf5, 0xfa }, { 0xf7, 0xbc }, { 0xf6, 0x7e },
    { 0xd9, 0x40 }, { 0xd8, 0x82 }, { 0xda, 0xc4 }, { 0xdb, 0x06 }, { 0xde, 0x48 }, { 0xdf, 0x8a }, { 0xdd, 0xcc }, { 0xdc, 0x0e },
    { 0xd7, 0x50 }, { 0xd6, 0x92 }, { 0xd4, 0xd4 }, { 0xd5, 0x16 }, { 0xd0, 0x58 }, { 0xd1, 0x9a }, { 0xd3, 0xdc }, { 0xd2, 0x1e },
    { 0xc5, 0x60 }, { 0xc4, 0xa2 }, { 0xc6, 0xe4 }, { 0xc7, 0x26 }, { 0xc2, 0x68 }, { 0xc3, 0xaa }, { 0xc1, 0xec }, { 0xc0, 0x2e },
    { 0xcb, 0x70 }, { 0xca, 0xb2 }, { 0xc8, 0xf4 }, { 0xc9, 0x36 }, { 0xcc, 0x78 }, { 0xcd, 0xba }, { 0xcf, 0xfc }, { 0xce, 0x3e },
    { 0x91, 0x80 }, { 0x90, 0x42 }, { 0x92, 0x04 }, { 0x93, 0xc6 }, { 0x96, 0x88 }, { 0x97, 0x4a }, { 0x95, 0x0c }, { 0x94, 0xce },
    { 0x9f, 0x90 }, { 0x9e, 0x52 }, { 0x9c, 0x14 }, { 0x9d, 0xd6 }, { 0x98, 0x98 }, { 0x99, 0x5a }, { 0x9b, 0x1c }, { 0x9a, 0xde },
    { 0x8d, 0xa0 }, { 0x8c, 0x62 }, { 0x8e, 0x24 }, { 0x8f, 0xe6 }, { 0x8a, 0xa8 }, { 0x8b, 0x6a }, { 0x89, 0x2c }, { 0x88, 0xee },
    { 0x83, 0xb0 }, { 0x82, 0x72 }, { 0x80, 0x34 }, { 0x81, 0xf6 }, { 0x84, 0xb8 }, { 0x85, 0x7a }, { 0x87, 0x3c }, { 0x86, 0xfe },
    { 0xa9, 0xc0 }, { 0xa8, 0x02 }, { 0xaa, 0x44 }, { 0xab, 0x86 }, { 0xae, 0xc8 }, { 0xaf, 0x0a }, { 0xad, 0x4c }, { 0xac, 0x8e },
    { 0xa7, 0xd0 }, { 0xa6, 0x12 }, { 0xa4, 0x54 }, { 0xa5, 0x96 }, { 0xa0, 0xd8 }, { 0xa1, 0x1a }, { 0xa3, 0x5c }, { 0xa2, 0x9e },
    { 0xb5, 0xe0 }, { 0xb4, 0x22 }, { 0xb6, 0x64 }, { 0xb7, 0xa6 }, { 0xb2, 0xe8 }, { 0xb3, 0x2a }, { 0xb1, 0x6c }, { 0xb0, 0xae },
    { 0xbb, 0xf0 }, { 0xba, 0x32 }, { 0xb8, 0x74 }, { 0xb9, 0xb6 }, { 0xbc, 0xf8 }, { 0xbd, 0x3a }, { 0xbf, 0x7c }, { 0xbe, 0xbe },
};
__host__ __device__  static void gcm_init_8bit_table(unsigned char hTable[][16], const unsigned char* h)
{
    unsigned char tmp[16];
    unsigned int i, j;

    memcpy(tmp, h, 16);
    memcpy(hTable[0x80], tmp, 16);

    for (i = 0x40; i >= 1; i >>= 1)
    {
        RSHIFT8x16_1(tmp);
        if (hTable[i << 1][15] & 1)
            tmp[0] ^= 0xe1;
        memcpy(hTable[i], tmp, 16);
    }

    for (i = 2; i < 256; i <<= 1)
    {
        for (j = 1; j < i; j++)
            XOR8x16(hTable[i + j], hTable[i], hTable[j]);
    }
}
__host__ __device__  static void gcm_gfmul_8(unsigned char* r, const unsigned char* x, const unsigned char hTable[][16])
{
    unsigned char z[16], mask;
    int i;

    memset(z, 0, 16);

    for (i = 15; i > 0; i--)
    {
        XOR8x16(z, z, hTable[x[i]]);

        mask = z[15];
        RSHIFT8x16_8(z);
        z[0] ^= reduction_8bit[mask][0];
        z[1] ^= reduction_8bit[mask][1];
    }

    XOR8x16(z, z, hTable[x[i]]);

    memcpy(r, z, 16);
}
__host__ __device__  static void _gcm_ghash_8(unsigned char* r, const unsigned char* x, unsigned int x_len, const unsigned char hTable[][16])
{
    unsigned int i;
    unsigned char y[16] = { 0, };

    memcpy(y, r, 16);

    for (; x_len >= 16; x_len -= 16, x += 16)
    {
        XOR8x16(y, y, x);

        gcm_gfmul_8(y, y, hTable);
    }

    if (x_len)
    {
        for (i = 0; i < x_len; i++)
            y[i] = y[i] ^ x[i];

        gcm_gfmul_8(y, y, hTable);
    }

    memcpy(r, y, 16);
}
__host__ __device__  static void _lea_gcm_init_8(LEA_GCM_CTX* ctx, const unsigned char* mk, int mk_len)
{
    unsigned char zero[16] = { 0, };

    memset(ctx, 0, sizeof(LEA_GCM_CTX));

    lea_set_key(&ctx->key, mk, mk_len);
    lea_encrypt(zero, zero, &ctx->key);

    gcm_init_8bit_table(ctx->h, zero);
}

//내 코드
__device__ void parallel_ghash1(uint8_t* dest, uint8_t* src1, uint8_t* src2, uint8_t* H_8) {
    uint8_t temp[16] = { 0x00, };
    gcm_gfmul_m(temp, src1, H_8);//->X_i*H^8
    XOR8x16(dest, temp, src2);//X_i *H^8  + X_(i+8)
    // __syncthreads();
}
__device__ void parallel_ghash2(uint8_t* dest, uint8_t* src1, uint8_t* src2, uint8_t* H_4) {
    uint8_t temp[16] = { 0x00, };
    gcm_gfmul_m(temp, src1, H_4);//->X_i*H^4
    XOR8x16(dest, temp, src2);//X_i *H^8  + X_(i+4)
    // __syncthreads();
}
__device__ void parallel_ghash3(uint8_t* dest, uint8_t* src1, uint8_t* src2, uint8_t* H_2) {
    uint8_t temp[16] = { 0x00, };
    gcm_gfmul_m(temp, src1, H_2);//->X_i*H^2
    XOR8x16(dest, temp, src2);//X_i *H^8  + X_(i+4)
}
__device__ void parallel_ghash_last(uint8_t* dest, uint8_t* src1, uint8_t* src2, uint8_t* H_2, uint8_t* H_1) {
    uint8_t temp1[16] = { 0x00, };
    uint8_t temp2[16] = { 0x00, };
    gcm_gfmul_m(temp1, src1, H_2);//->X_i*H^2
    gcm_gfmul_m(temp2, src2, H_1);//->X_i*H
    XOR8x16(dest, temp1, temp2);//X_i *H^8  + X_(i+4)
}
//GCM_REF를 위한 함수들임
__host__ __device__ static void ctr128_inc(unsigned char* counter) {
    unsigned int n = 16;
    unsigned char c;

    do {
        --n;
        c = counter[n];
        ++c;
        counter[n] = c;
        if (c) return;
    } while (n);
}
__host__ __device__ static void ctr128_inc_aligned(unsigned char* counter) {
    unsigned int* data, c, n;
    const union { long one; char little; } is_endian = { 1 };

    if (is_endian.little) {
        ctr128_inc(counter);
        return;
    }

    data = (unsigned int*)counter;
    n = 16 / sizeof(unsigned int);
    do {
        --n;
        c = data[n];
        ++c;
        data[n] = c;
        if (c) return;
    } while (n);
}
__host__ __device__ void ctr_enc(unsigned char* ct, const unsigned char* pt, unsigned int pt_len, unsigned char* ctr, const LEA_KEY* key)
{
    unsigned char block[16] = { 0x00, };

    if (!ctr || !key || pt_len == 0) {
        return;
    }

    unsigned int numBlocks = pt_len >> 4;

    for (unsigned int i = 0; i < numBlocks; i++, pt += 16, ct += 16) {
        lea_encrypt(block, ctr, key);
        XOR8x16(ct, block, pt);
        ctr128_inc_aligned(ctr);
    }

    if (pt_len & 0xF) {
        lea_encrypt(block, ctr, key);
        for (unsigned int i = 0; i < (pt_len & 0xF); i++) {
            ct[i] = block[i] ^ pt[i];
        }
    }
}
__host__ __device__ void ctr_dec(unsigned char* pt, const unsigned char* ct, unsigned int ct_len, unsigned char* ctr, const LEA_KEY* key)
{
    ctr_enc(pt, ct, ct_len, ctr, key);
}
__host__ __device__ void gcm_set_ctr(LEA_GCM_CTX* ctx, const unsigned char* iv, int iv_len)
{
    int tmp_iv_len = iv_len;

    if (!ctx || !iv) {
        return;
    }
    if (iv_len < 0) {
        return;
    }

    ctx->ct_len = 0;

    if (iv_len == 12)
    {
        memcpy(ctx->ctr, iv, 12);
        ctx->ctr[15] = 1;
    }
    else
    {
        for (; iv_len >= 16; iv_len -= 16, iv += 16)
            _gcm_ghash_m(ctx->ctr, iv, 16, (const unsigned char(*)[16])ctx->h);

        if (iv_len)
            _gcm_ghash_m(ctx->ctr, iv, iv_len, (const unsigned char(*)[16])ctx->h);

        tmp_iv_len <<= 3;
        ctx->yn[12] = (tmp_iv_len >> 24) & 0xff;
        ctx->yn[13] = (tmp_iv_len >> 16) & 0xff;
        ctx->yn[14] = (tmp_iv_len >> 8) & 0xff;
        ctx->yn[15] = (tmp_iv_len) & 0xff;
        _gcm_ghash_m(ctx->ctr, ctx->yn, 16, (const unsigned char(*)[16])ctx->h);
        memset(ctx->yn, 0, 16);
    }

    lea_encrypt(ctx->ek0, ctx->ctr, &ctx->key);

    ctr128_inc_aligned(ctx->ctr);
}
__host__ __device__ void gcm_set_aad(LEA_GCM_CTX* ctx, const unsigned char* aad, int aad_len)
{
    if (!ctx) {
        return;
    }
    if (aad_len <= 0) {
        return;
    }
    if (!aad) {
        return;
    }
    ctx->aad_len = aad_len;

    _gcm_ghash_m(ctx->tbl, aad, aad_len, (const unsigned char(*)[16])ctx->h);
}
__host__ __device__ void gcm_enc(LEA_GCM_CTX* ctx, unsigned char* ct, const unsigned char* pt, int pt_len)
{
    int remain, i;

    if (!ctx || !ct || !pt) {
        return;
    }
    if (pt_len < 0) {
        return;
    }

    ctx->is_encrypt = 1;
    ctx->ct_len += pt_len;

    if (!pt_len)
        return;

    if (ctx->yn_used)
    {
        remain = 16 - ctx->yn_used;

        if (remain > pt_len)
            remain = pt_len;

        for (i = 0; i < remain; i++)
            ctx->yn[ctx->yn_used + i] ^= pt[i];

        memcpy(ct, ctx->yn + ctx->yn_used, remain);

        pt_len -= remain;
        pt += remain;
        ct += remain;
        ctx->yn_used = (ctx->yn_used + remain) & 0xf;

        if (!ctx->yn_used) {
            _gcm_ghash_m(ctx->tbl, ctx->yn, 16, (const unsigned char(*)[16])ctx->h);
        }


        if (!pt_len)
            return;
    }

    i = pt_len & 0xfffffff0;

    ctr_enc(ct, pt, i, ctx->ctr, &ctx->key);

    if (i) {
        _gcm_ghash_m(ctx->tbl, ct, i, (const unsigned char(*)[16])ctx->h);
    }


    pt_len &= 0xf;

    if (!pt_len)
        return;

    pt += i;
    ct += i;

    lea_encrypt(ctx->yn, ctx->ctr, &ctx->key);
    ctr128_inc_aligned(ctx->ctr);
    ctx->yn_used = pt_len;

    for (pt_len--; pt_len >= 0; pt_len--)
        ct[pt_len] = ctx->yn[pt_len] = ctx->yn[pt_len] ^ pt[pt_len];
}
__host__ __device__ int gcm_final(LEA_GCM_CTX* ctx, unsigned char* tag, int tag_len)
{
    unsigned char tmp[16];

    if (!ctx || !tag) {
        memset(ctx, 0, sizeof(LEA_GCM_CTX));
        return -1;
    }
    if (tag_len < 4) {
        memset(ctx, 0, sizeof(LEA_GCM_CTX));
        return -1;
    }

    if (ctx->yn_used)
        _gcm_ghash_m(ctx->tbl, ctx->yn, ctx->yn_used, (const unsigned char(*)[16])ctx->h);

    memset(tmp, 0, 16);

    ctx->aad_len <<= 3;
    ctx->ct_len <<= 3;

    tmp[4] = ctx->aad_len >> 24;
    tmp[5] = ctx->aad_len >> 16;
    tmp[6] = ctx->aad_len >> 8;
    tmp[7] = ctx->aad_len;

    tmp[12] = ctx->ct_len >> 24;
    tmp[13] = ctx->ct_len >> 16;
    tmp[14] = ctx->ct_len >> 8;
    tmp[15] = ctx->ct_len;
    _gcm_ghash_m(ctx->tbl, tmp, 16, (const unsigned char(*)[16])ctx->h);
    XOR8x16(ctx->tbl, ctx->tbl, ctx->ek0);
    if (ctx->is_encrypt)
        memcpy(tag, ctx->tbl, tag_len);
    else
    {
        for (tag_len--; tag_len >= 0; tag_len--)
        {
            if (ctx->tbl[tag_len] != tag[tag_len]) {
                memset(ctx->ctr, 0, 16);
                memset(ctx->ek0, 0, 16);
                memset(ctx->tbl, 0, 16);
                memset(ctx->yn, 0, 16);
                ctx->yn_used = 0;

                return -1;

            }

        }
    }
    ctx->ct_len = 0;

    memset(ctx->ctr, 0, 16);
    memset(ctx->ek0, 0, 16);
    memset(ctx->tbl, 0, 16);
    memset(ctx->yn, 0, 16);
    ctx->yn_used = 0;

    return 0;
}

//4bit table의 GCM
__host__ __device__ void gcm_set_ctr_4(LEA_GCM_CTX* ctx, const unsigned char* iv, int iv_len)
{

    int tmp_iv_len = iv_len;

    if (!ctx || !iv) {
        return;
    }
    if (iv_len < 0) {
        return;
    }

    ctx->ct_len = 0;

    if (iv_len == 12)
    {
        memcpy(ctx->ctr, iv, 12);
        ctx->ctr[15] = 1;
    }
    else
    {
        for (; iv_len >= 16; iv_len -= 16, iv += 16)
            _gcm_ghash_4(ctx->ctr, iv, 16, (const unsigned char(*)[16])ctx->h);

        if (iv_len)
            _gcm_ghash_4(ctx->ctr, iv, iv_len, (const unsigned char(*)[16])ctx->h);

        tmp_iv_len <<= 3;
        ctx->yn[12] = (tmp_iv_len >> 24) & 0xff;
        ctx->yn[13] = (tmp_iv_len >> 16) & 0xff;
        ctx->yn[14] = (tmp_iv_len >> 8) & 0xff;
        ctx->yn[15] = (tmp_iv_len) & 0xff;
        _gcm_ghash_4(ctx->ctr, ctx->yn, 16, (const unsigned char(*)[16])ctx->h);
        memset(ctx->yn, 0, 16);
    }

    lea_encrypt(ctx->ek0, ctx->ctr, &ctx->key);

    ctr128_inc_aligned(ctx->ctr);
}
__host__ __device__ void gcm_set_aad_4(LEA_GCM_CTX* ctx, const unsigned char* aad, int aad_len)
{
    if (!ctx) {
        return;
    }
    if (!aad) {
        return;
    }
    if (aad_len <= 0) {
        return;
    }

    ctx->aad_len = aad_len;


    _gcm_ghash_4(ctx->tbl, aad, aad_len, (const unsigned char(*)[16])ctx->h);
}
__host__ __device__ void gcm_enc_4(LEA_GCM_CTX* ctx, unsigned char* ct, const unsigned char* pt, int pt_len)
{
    int remain, i;

    if (!ctx || !ct || !pt) {
        return;
    }
    if (pt_len < 0) {
        return;
    }

    ctx->is_encrypt = 1;
    ctx->ct_len += pt_len;

    if (!pt_len)
        return;

    if (ctx->yn_used)
    {
        remain = 16 - ctx->yn_used;

        if (remain > pt_len)
            remain = pt_len;

        for (i = 0; i < remain; i++)
            ctx->yn[ctx->yn_used + i] ^= pt[i];

        memcpy(ct, ctx->yn + ctx->yn_used, remain);

        pt_len -= remain;
        pt += remain;
        ct += remain;
        ctx->yn_used = (ctx->yn_used + remain) & 0xf;

        if (!ctx->yn_used)
            _gcm_ghash_4(ctx->tbl, ctx->yn, 16, (const unsigned char(*)[16])ctx->h);

        if (!pt_len)
            return;
    }

    i = pt_len & 0xfffffff0;

    ctr_enc(ct, pt, i, ctx->ctr, &ctx->key);

    if (i)
        _gcm_ghash_4(ctx->tbl, ct, i, (const unsigned char(*)[16])ctx->h);

    pt_len &= 0xf;

    if (!pt_len)
        return;

    pt += i;
    ct += i;

    lea_encrypt(ctx->yn, ctx->ctr, &ctx->key);
    ctr128_inc_aligned(ctx->ctr);
    ctx->yn_used = pt_len;

    for (pt_len--; pt_len >= 0; pt_len--)
        ct[pt_len] = ctx->yn[pt_len] = ctx->yn[pt_len] ^ pt[pt_len];
}
__host__ __device__ void gcm_dec_4(LEA_GCM_CTX* ctx, unsigned char* pt, const unsigned char* ct, int ct_len)
{
    int remain, i;

    if (!ctx || !pt || !ct) {
        return;
    }
    if (ct_len < 0) {
        return;
    }

    ctx->is_encrypt = 0;
    ctx->ct_len += ct_len;

    if (!ct_len)
        return;

    if (ctx->yn_used)
    {
        remain = 16 - ctx->yn_used;

        if (remain > ct_len)
            remain = ct_len;

        for (i = 0; i < remain; i++)
            pt[i] = ctx->yn[ctx->yn_used + i] ^ ct[i];

        memcpy(ctx->yn + ctx->yn_used, ct, remain);

        ct_len -= remain;
        pt += remain;
        ct += remain;
        ctx->yn_used = (ctx->yn_used + remain) & 0xf;

        if (!ctx->yn_used)
            _gcm_ghash_4(ctx->tbl, ctx->yn, 16, (const unsigned char(*)[16])ctx->h);

        if (!ct_len)
            return;
    }

    i = ct_len & 0xfffffff0;

    ctr_dec(pt, ct, i, ctx->ctr, &ctx->key);

    if (i)
        _gcm_ghash_4(ctx->tbl, ct, i, (const unsigned char(*)[16])ctx->h);

    ct_len &= 0xf;

    if (!ct_len)
        return;

    pt += i;
    ct += i;

    lea_encrypt(ctx->yn, ctx->ctr, &ctx->key);
    ctr128_inc_aligned(ctx->ctr);
    ctx->yn_used = ct_len;

    for (ct_len--; ct_len >= 0; ct_len--) {
        pt[ct_len] = ctx->yn[ct_len] ^ ct[ct_len];
        ctx->yn[ct_len] = ct[ct_len];
    }
}
__host__ __device__ int gcm_final_4(LEA_GCM_CTX* ctx, unsigned char* tag, int tag_len)
{
    unsigned char tmp[16];

    if (!ctx || !tag) {
        memset(ctx, 0, sizeof(LEA_GCM_CTX));
        return -1;
    }
    if (tag_len < 4) {
        memset(ctx, 0, sizeof(LEA_GCM_CTX));
        return -1;
    }

    if (ctx->yn_used)
        _gcm_ghash_4(ctx->tbl, ctx->yn, ctx->yn_used, (const unsigned char(*)[16])ctx->h);

    memset(tmp, 0, 16);

    ctx->aad_len <<= 3;
    ctx->ct_len <<= 3;

    tmp[4] = ctx->aad_len >> 24;
    tmp[5] = ctx->aad_len >> 16;
    tmp[6] = ctx->aad_len >> 8;
    tmp[7] = ctx->aad_len;

    tmp[12] = ctx->ct_len >> 24;
    tmp[13] = ctx->ct_len >> 16;
    tmp[14] = ctx->ct_len >> 8;
    tmp[15] = ctx->ct_len;

    _gcm_ghash_4(ctx->tbl, tmp, 16, (const unsigned char(*)[16])ctx->h);

    XOR8x16(ctx->tbl, ctx->tbl, ctx->ek0);

    if (ctx->is_encrypt)
        memcpy(tag, ctx->tbl, tag_len);
    else
    {
        for (tag_len--; tag_len >= 0; tag_len--)
        {
            if (ctx->tbl[tag_len] != tag[tag_len]) {
                memset(ctx->ctr, 0, 16);
                memset(ctx->ek0, 0, 16);
                memset(ctx->tbl, 0, 16);
                memset(ctx->yn, 0, 16);
                ctx->yn_used = 0;

                return -1;

            }

        }
    }
    ctx->ct_len = 0;

    memset(ctx->ctr, 0, 16);
    memset(ctx->ek0, 0, 16);
    memset(ctx->tbl, 0, 16);
    memset(ctx->yn, 0, 16);
    ctx->yn_used = 0;

    return 0;
}
//8bit table version GCM
__host__ __device__ void gcm_set_ctr_8(LEA_GCM_CTX* ctx, const unsigned char* iv, int iv_len)
{
    int tmp_iv_len = iv_len;

    if (!ctx || !iv) {
        return;
    }
    if (iv_len < 0) {
        return;
    }

    ctx->ct_len = 0;

    if (iv_len == 12)
    {
        memcpy(ctx->ctr, iv, 12);
        ctx->ctr[15] = 1;
    }
    else
    {
        for (; iv_len >= 16; iv_len -= 16, iv += 16)
            _gcm_ghash_8(ctx->ctr, iv, 16, (const unsigned char(*)[16])ctx->h);

        if (iv_len)
            _gcm_ghash_8(ctx->ctr, iv, iv_len, (const unsigned char(*)[16])ctx->h);

        tmp_iv_len <<= 3;
        ctx->yn[12] = (tmp_iv_len >> 24) & 0xff;
        ctx->yn[13] = (tmp_iv_len >> 16) & 0xff;
        ctx->yn[14] = (tmp_iv_len >> 8) & 0xff;
        ctx->yn[15] = (tmp_iv_len) & 0xff;
        _gcm_ghash_8(ctx->ctr, ctx->yn, 16, (const unsigned char(*)[16])ctx->h);
        memset(ctx->yn, 0, 16);
    }

    lea_encrypt(ctx->ek0, ctx->ctr, &ctx->key);

    ctr128_inc_aligned(ctx->ctr);
}
__host__ __device__ void gcm_set_aad_8(LEA_GCM_CTX* ctx, const unsigned char* aad, int aad_len)
{
    if (!ctx) {
        return;
    }
    if (aad_len <= 0) {
        return;
    }
    if (!aad) {
        return;
    }
    ctx->aad_len = aad_len;

    _gcm_ghash_8(ctx->tbl, aad, aad_len, (const unsigned char(*)[16])ctx->h);
}
__host__ __device__ void gcm_enc_8(LEA_GCM_CTX* ctx, unsigned char* ct, const unsigned char* pt, int pt_len) {
    int remain, i;

    if (!ctx || !ct || !pt) {
        return;
    }
    if (pt_len < 0) {
        return;
    }

    ctx->is_encrypt = 1;
    ctx->ct_len += pt_len;

    if (!pt_len) {
        printf("\nreturn\n");
        return;
    }

    if (ctx->yn_used)
    {
        remain = 16 - ctx->yn_used;

        if (remain > pt_len)
            remain = pt_len;

        for (i = 0; i < remain; i++)
            ctx->yn[ctx->yn_used + i] ^= pt[i];

        memcpy(ct, ctx->yn + ctx->yn_used, remain);

        pt_len -= remain;
        pt += remain;
        ct += remain;
        ctx->yn_used = (ctx->yn_used + remain) & 0xf;

        if (!ctx->yn_used)
            _gcm_ghash_8(ctx->tbl, ctx->yn, 16, (const unsigned char(*)[16])ctx->h);

        if (!pt_len)
            return;
    }

    i = pt_len & 0xfffffff0;

    ctr_enc(ct, pt, i, ctx->ctr, &ctx->key);

    if (i)
        _gcm_ghash_8(ctx->tbl, ct, i, (const unsigned char(*)[16])ctx->h);

    pt_len &= 0xf;

    if (!pt_len)
        return;

    pt += i;
    ct += i;

    lea_encrypt(ctx->yn, ctx->ctr, &ctx->key);
    ctr128_inc_aligned(ctx->ctr);
    ctx->yn_used = pt_len;

    for (pt_len--; pt_len >= 0; pt_len--)
        ct[pt_len] = ctx->yn[pt_len] = ctx->yn[pt_len] ^ pt[pt_len];
}
__host__ __device__ void gcm_dec_8(LEA_GCM_CTX* ctx, unsigned char* pt, const unsigned char* ct, int ct_len)
{
    int remain, i;

    if (!ctx || !pt || !ct) {
        return;
    }
    if (ct_len < 0) {
        return;
    }

    ctx->is_encrypt = 0;
    ctx->ct_len += ct_len;

    if (!ct_len)
        return;

    if (ctx->yn_used)
    {
        remain = 16 - ctx->yn_used;

        if (remain > ct_len)
            remain = ct_len;

        for (i = 0; i < remain; i++)
            pt[i] = ctx->yn[ctx->yn_used + i] ^ ct[i];

        memcpy(ctx->yn + ctx->yn_used, ct, remain);

        ct_len -= remain;
        pt += remain;
        ct += remain;
        ctx->yn_used = (ctx->yn_used + remain) & 0xf;

        if (!ctx->yn_used)
            _gcm_ghash_8(ctx->tbl, ctx->yn, 16, (const unsigned char(*)[16])ctx->h);

        if (!ct_len)
            return;
    }

    i = ct_len & 0xfffffff0;

    ctr_dec(pt, ct, i, ctx->ctr, &ctx->key);

    if (i)
        _gcm_ghash_8(ctx->tbl, ct, i, (const unsigned char(*)[16])ctx->h);

    ct_len &= 0xf;

    if (!ct_len)
        return;

    pt += i;
    ct += i;

    lea_encrypt(ctx->yn, ctx->ctr, &ctx->key);
    ctr128_inc_aligned(ctx->ctr);
    ctx->yn_used = ct_len;

    for (ct_len--; ct_len >= 0; ct_len--) {
        pt[ct_len] = ctx->yn[ct_len] ^ ct[ct_len];
        ctx->yn[ct_len] = ct[ct_len];
    }
}
__host__ __device__ int gcm_final_8(LEA_GCM_CTX* ctx, unsigned char* tag, int tag_len)
{
    unsigned char tmp[16];

    if (!ctx || !tag) {
        memset(ctx, 0, sizeof(LEA_GCM_CTX));
        return -1;
    }
    if (tag_len < 4) {
        memset(ctx, 0, sizeof(LEA_GCM_CTX));
        return -1;
    }

    if (ctx->yn_used)
        _gcm_ghash_8(ctx->tbl, ctx->yn, ctx->yn_used, (const unsigned char(*)[16])ctx->h);

    memset(tmp, 0, 16);

    ctx->aad_len <<= 3;
    ctx->ct_len <<= 3;

    tmp[4] = ctx->aad_len >> 24;
    tmp[5] = ctx->aad_len >> 16;
    tmp[6] = ctx->aad_len >> 8;
    tmp[7] = ctx->aad_len;

    tmp[12] = ctx->ct_len >> 24;
    tmp[13] = ctx->ct_len >> 16;
    tmp[14] = ctx->ct_len >> 8;
    tmp[15] = ctx->ct_len;

    _gcm_ghash_8(ctx->tbl, tmp, 16, (const unsigned char(*)[16])ctx->h);

    XOR8x16(ctx->tbl, ctx->tbl, ctx->ek0);

    if (ctx->is_encrypt) {
        memcpy(tag, ctx->tbl, tag_len);
    }

    else
    {
        for (tag_len--; tag_len >= 0; tag_len--)
        {
            if (ctx->tbl[tag_len] != tag[tag_len]) {
                memset(ctx->ctr, 0, 16);
                memset(ctx->ek0, 0, 16);
                memset(ctx->tbl, 0, 16);
                memset(ctx->yn, 0, 16);
                ctx->yn_used = 0;

                return -1;

            }

        }
    }
    ctx->ct_len = 0;

    memset(ctx->ctr, 0, 16);
    memset(ctx->ek0, 0, 16);
    memset(ctx->tbl, 0, 16);
    memset(ctx->yn, 0, 16);
    ctx->yn_used = 0;

    return 0;
}



__global__ void parallel_enc_GHASH(uint8_t* tag, uint8_t* aad, uint8_t* ctr, uint8_t* pt, uint8_t* H, LEA_KEY* key, uint8_t* Y) {
    /*
    iv[12]
    pt[128]
    ctr[16 * 7] //ctr0 = Y로 바꿀것 , ctr1,...,ctr7-> 카운터들   인덱스 조절이 필요함
    aad[16 * 16]//앞은 aad[8][16] / enc(counter) // len으로 채울 것임
    mk[16]
    tag[16]
    */

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //병렬화 암호화 코드
    lea_encrypt(aad + (tid / 8) * 16 * 16 + 7 * 16 + (tid % 8) * 16, ctr + tid * 16, key);
    XOR8x16(aad + (tid / 8) * 16 * 16 + 7 * 16 + (tid % 8) * 16, pt + tid * 16, aad + (tid / 8) * 16 * 16 + 7 * 16 + (tid % 8) * 16);

    //병렬화 GHASH코드
    parallel_ghash1(ctr + tid * 16, aad + (tid / 8) * 16 * 16 + (tid % 8) * 16, aad + (tid / 8) * 16 * 16 + (tid % 8) * 16 + 8 * 16, H + 3 * 16);
    parallel_ghash2(aad + tid * 16, ctr + (tid / 4) * 8 * 16 + (tid % 4) * 16, ctr + (tid / 4) * 8 * 16 + (tid % 4) * 16 + 4 * 16, H + 2 * 16);
    parallel_ghash3(ctr + tid * 16, aad + (tid / 2) * 4 * 16 + (tid % 2) * 16, aad + (tid / 2) * 4 * 16 + (tid % 2) * 16 + 2 * 16, H + 1 * 16);
    parallel_ghash_last(aad + tid * 16, ctr + tid * 2 * 16, ctr + tid * 2 * 16 + 16, H + 1 * 16, H);

    //tag XOR
    XOR8x16(tag + tid * 16, Y, aad + tid * 16);

}

//#define Alg_num 4           //1 block 8 thread = 1 alg, 1 block 32 thread = 4 alg
void test_GPU_parallel(int BlockSize, int ThreadSize) {//이게 약간 main문 같은것 다 박으면 됨
    int mk_len = 16; //klen
    int iv_len = 12; //nlen
    int pt_len = 16 * 8;	//plen
    int aad_len = 112;//alen
    int tag_len = 16;//tlen
    int ct_len = pt_len;

    int Alg_num = ThreadSize / 4;

    uint8_t iv[12] = {
        0x67,0x61,0x32,0x6D,0xE2,0x80,0x63,0x78,0x2E,0x96,0x72,0x98
    }; // 넘기는 값이 아님->Counter 값 사전 계산을 위함
    uint8_t cpu_pt[16 * 8] = {
        0x29,0xDA,0x4D,0x54,0x4C,0xAC,0x60,0xB8,0x83,0x1E,0x0A,0xFB,0x3B,0xB4,0x4E,0x5F,
        0xBC,0x68,0xCC,0x59,0xB9,0xF1,0xEF,0xF2,0x25,0x45,0x67,0x6D,0x49,0x5D,0xDA,0x2A,
        0x14,0x38,0xD6,0xCA,0xB2,0x22,0x0B,0x94,0x60,0x36,0xB7,0x17,0x7E,0x22,0x61,0x11,
        0xE5,0x2A,0xCA,0x90,0x7C,0x70,0x21,0x57,0x06,0x72,0x76,0x83,0x3E,0xD4,0x71,0x6F,
        0x26,0x60,0x44,0xD4,0x9C,0x4B,0xDE,0x35,0x2E,0xB9,0x61,0x7A,0x2F,0x84,0xD7,0xDB,
        0x0A,0x39,0x21,0xFF,0xD7,0x64,0x2F,0x65,0x2C,0x0E,0x77,0x04,0x36,0x83,0x9F,0x2E,
        0x08,0x59,0x7D,0xBA,0x32,0xAD,0x42,0x62,0x96,0xB0,0xF2,0x6A,0x77,0x63,0x18,0x83,
        0xD3,0x9E,0xB7,0xEE,0xF3,0x57,0x83,0xE4,0x40,0x21,0x51,0x36,0x58,0xF3,0xCD,0x31

    };
    uint8_t* pt = NULL;
    pt = (uint8_t*)malloc(sizeof(uint8_t) * BlockSize * Alg_num * 16 * 8);
    for (int i = 0; i < BlockSize * Alg_num; i++) {
        memcpy(pt + 16 * 8 * i, cpu_pt, sizeof(uint8_t) * 16 * 8);
    }
    uint8_t cpu_aad[16 * 16] = {  //앞은 aad / enc(counter) // len으로 채울 것임
        0xE7,0x60,0x76,0xA5,0xF7,0xD0,0x84,0x9C,0xE9,0xB4,0xEE,0x62,0xCD,0xAB,0x61,0x3E,
        0xDA,0xDF,0xF5,0x3A,0x05,0x42,0x5E,0x84,0xD3,0x17,0x66,0x14,0x6B,0xB5,0x9B,0x8F,
        0xEF,0x87,0x44,0x6B,0x0C,0x58,0x9A,0x55,0xD0,0xCD,0x35,0xCF,0xBC,0x33,0xC5,0x1E,
        0x1B,0x7B,0x02,0x63,0x92,0x83,0xE6,0x08,0xF7,0x98,0x19,0x5D,0x66,0x38,0xAD,0x83,
        0x92,0x1C,0x0B,0x8A,0x5A,0x76,0x2D,0xD6,0x9D,0x77,0x0E,0xC7,0x4E,0x14,0x5F,0x99,
        0xCB,0x0A,0x7C,0x88,0x6A,0x1F,0x9E,0x38,0x55,0xAB,0x52,0x55,0x99,0x58,0x2A,0x76,
        0x79,0x26,0xB2,0x48,0x08,0xB2,0xA5,0xE8,0xF7,0xB7,0x61,0x90,0xAB,0xD5,0x99,0x76,
        0x00,
    };
    uint8_t* aad = NULL;
    aad = (uint8_t*)malloc(sizeof(uint8_t) * BlockSize * Alg_num * 16 * 16);

    // len값 저장
    aad_len <<= 3;
    ct_len <<= 3;
    cpu_aad[16 * 15 + 4] = aad_len >> 24;
    cpu_aad[16 * 15 + 5] = aad_len >> 16;
    cpu_aad[16 * 15 + 6] = aad_len >> 8;
    cpu_aad[16 * 15 + 7] = aad_len;
    cpu_aad[16 * 15 + 12] = ct_len >> 24;
    cpu_aad[16 * 15 + 13] = ct_len >> 16;
    cpu_aad[16 * 15 + 14] = ct_len >> 8;
    cpu_aad[16 * 15 + 15] = ct_len;
    //printf("hi\n");
    for (int i = 0; i < BlockSize * Alg_num; i++) {
        memcpy(aad + 16 * 16 * i, cpu_aad, sizeof(uint8_t) * 16 * 16);
        //print_hex(aad + 16*16 * i, 16*16);
    }
    uint8_t mk[16] = {
        0x43,0x60,0x77,0xD9,0xEF,0x6A,0x74,0xDC,0x3F,0xB2,0x37,0xFC,0xE6,0xEB,0x3D,0x11
    };
    //lea_encrypt(h, zero, mk);
    uint8_t cpu_tag[16] = {
        0x00,
    };
    uint8_t* tag = NULL;
    tag = (uint8_t*)malloc(sizeof(uint8_t) * BlockSize * Alg_num * 16);
    for (int i = 0; i < BlockSize * Alg_num; i++) {
        memcpy(tag + 16 * i, cpu_tag, sizeof(uint8_t) * 16);
    }

    //key 만들기   
    LEA_KEY* key;
    key = (LEA_KEY*)malloc(sizeof(LEA_KEY) * BlockSize * Alg_num);
    memset(key, 0, sizeof(LEA_KEY) * BlockSize * Alg_num);
    lea_set_key(key, mk, mk_len);
    LEA_KEY* g_key = NULL;
    cudaMalloc((void**)&g_key, sizeof(LEA_KEY));


    //H,Y제작
    uint8_t ZERO[16] = { 0x00, };
    uint8_t* H = NULL;
    H = (uint8_t*)malloc(sizeof(uint8_t) * 16 * 4);
    lea_encrypt(H, ZERO, key);
    gcm_gfmul_m(H + 16, H, H);  //--> ctx->sub_h[1] = H^2
    gcm_gfmul_m(H + 16 * 2, H + 16, H + 16);  //--> ctx->sub_h[2] = H^4
    gcm_gfmul_m(H + 16 * 3, H + 16 * 2, H + 16 * 2);  //--> ctx->sub_h[3] = H^8

    //Y제작
    uint8_t cpu_Y[16] = { 0x00, };
    uint8_t* Y = NULL;
    Y = (uint8_t*)malloc(sizeof(uint8_t) * BlockSize * Alg_num * 16);
    memcpy(cpu_Y, iv, 12);
    cpu_Y[15] = 1;
    lea_encrypt(cpu_Y, cpu_Y, key);
    for (int i = 0; i < BlockSize * Alg_num; i++) {
        memcpy(Y + 16 * i, cpu_Y, 16);
    }
    uint8_t* ctr = NULL;
    ctr = (uint8_t*)malloc(sizeof(uint8_t) * BlockSize * Alg_num * 16 * 8);
    memset(ctr, 0, sizeof(uint8_t) * BlockSize * Alg_num * 16 * 8);
    memcpy(&ctr[0], iv, 12);//CTR0값
    ctr[15] = 2;
    memcpy(&ctr[1 << 4], iv, 12);//CTR1값
    ctr[(1 << 4) + 15] = 3;
    memcpy(&ctr[2 << 4], iv, 12);//CTR2값
    ctr[(2 << 4) + 15] = 4;
    memcpy(&ctr[3 << 4], iv, 12);//CTR3값
    ctr[(3 << 4) + 15] = 5;
    memcpy(&ctr[4 << 4], iv, 12);//CTR4값
    ctr[(4 << 4) + 15] = 6;
    memcpy(&ctr[5 << 4], iv, 12);//CTR5값
    ctr[(5 << 4) + 15] = 7;
    memcpy(&ctr[6 << 4], iv, 12);//CTR6값
    ctr[(6 << 4) + 15] = 8;
    memcpy(&ctr[7 << 4], iv, 12);//CTR6값
    ctr[(7 << 4) + 15] = 9;
    for (int i = 0; i < BlockSize * Alg_num; i++) {
        memcpy(ctr + 16 * 8 * i, ctr, sizeof(uint8_t) * 16 * 8);
    }
    //넘겨 줘야할 값들(g_tag, g_aad, g_ctr, g_pt, &g_key,g_H)
    //Y
    uint8_t* g_Y = NULL;
    cudaMalloc((void**)&g_Y, BlockSize * Alg_num * 16 * sizeof(uint8_t));
    //H
    uint8_t* g_H = NULL;
    cudaMalloc((void**)&g_H, BlockSize * Alg_num * 16 * 4 * sizeof(uint8_t));
    // pt
    uint8_t* g_pt = NULL;
    cudaMalloc((void**)&g_pt, BlockSize * Alg_num * 16 * 8 * sizeof(uint8_t));
    // ctr -> 암호화 값 저장
    uint8_t* g_ctr = NULL;
    cudaMalloc((void**)&g_ctr, BlockSize * Alg_num * 16 * 8 * sizeof(uint8_t));
    // aad //len값 마지막에 포함
    uint8_t* g_aad = NULL;
    cudaMalloc((void**)&g_aad, BlockSize * Alg_num * 16 * 16 * sizeof(uint8_t));
    //tag 결과값 저장 
    uint8_t* g_tag = NULL;
    cudaMalloc((void**)&g_tag, BlockSize * Alg_num * 16 * sizeof(uint8_t));



    //cuda에 값 복제
    cudaMemcpy((void**)g_H, H, BlockSize * Alg_num * 16 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_pt, pt, BlockSize * Alg_num * 16 * 8, cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_ctr, ctr, BlockSize * Alg_num * 16 * 8, cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_key, key, sizeof(LEA_KEY), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_tag, tag, BlockSize * Alg_num * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_aad, aad, BlockSize * Alg_num * 16 * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_Y, Y, BlockSize * Alg_num * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    //성능측정
    cudaEvent_t start, stop;
    float elapsed_time_ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    //Global
    parallel_enc_GHASH << < (BlockSize + 31) / 32, ThreadSize >> > (g_tag, g_aad, g_ctr, g_pt, g_H, g_key, g_Y);


    cudaMemcpy(aad, g_aad, BlockSize * Alg_num * sizeof(uint8_t) * 16 * 16, cudaMemcpyDeviceToHost);

    


    cudaDeviceSynchronize();
    //성능측정
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    printf("my code\n");
    printf("elapsed_time_ms is %4.4f\n", elapsed_time_ms);
    printf("Performance : %4.2f GCM time per second \n", BlockSize * Alg_num / ((elapsed_time_ms / 1000)));
    //cuda->cpu
    cudaMemcpy(tag, g_tag, BlockSize * Alg_num * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(g_pt);
    cudaFree(g_tag);
    cudaFree(g_aad);
    cudaFree(g_ctr);
    cudaFree(g_H);
    cudaFree(g_Y);
    cudaFree(g_key);

    free(pt);
    free(ctr);
    free(aad);
    free(H);
    free(key);
    free(tag);
    free(Y);
}
__global__ void GCM_REF(uint8_t* mk, int mk_len, uint8_t* iv, int iv_len, uint8_t* aad, int aad_len, uint8_t* ct, uint8_t* pt, int pt_len, uint8_t* tag, int tag_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    LEA_GCM_CTX ctx;
    _lea_gcm_init_m(&ctx, mk + tid * 16, mk_len);
    gcm_set_ctr(&ctx, iv + tid * 12, iv_len);
    gcm_set_aad(&ctx, aad + tid * 112, aad_len);
    gcm_enc(&ctx, ct + tid * 128, pt + tid * 128, pt_len);
    gcm_final(&ctx, tag + tid * 16, tag_len);
}
void test_GPU_REF(int BlockSize, int ThreadSize) {//이게 약간 main문 같은것 다 박으면 됨
    //ctx
   // LEA_GCM_CTX* ctx;
    //ctx = (LEA_GCM_CTX*)malloc(BlockSize * ThreadSize * sizeof(LEA_GCM_CTX));
    //iv
    uint8_t cpu_iv[12] = {
        0x67,0x61,0x32,0x6D,0xE2,0x80,0x63,0x78,0x2E,0x96,0x72,0x98
    };
    uint8_t* iv = NULL;
    iv = (uint8_t*)malloc(BlockSize * ThreadSize * 12 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(iv + 12 * i, cpu_iv, 12 * sizeof(uint8_t));
    }
    //pt
    uint8_t cpu_pt[128] = {
        0x29,0xDA,0x4D,0x54,0x4C,0xAC,0x60,0xB8,0x83,0x1E,0x0A,0xFB,0x3B,0xB4,0x4E,0x5F,
        0xBC,0x68,0xCC,0x59,0xB9,0xF1,0xEF,0xF2,0x25,0x45,0x67,0x6D,0x49,0x5D,0xDA,0x2A,
        0x14,0x38,0xD6,0xCA,0xB2,0x22,0x0B,0x94,0x60,0x36,0xB7,0x17,0x7E,0x22,0x61,0x11,
        0xE5,0x2A,0xCA,0x90,0x7C,0x70,0x21,0x57,0x06,0x72,0x76,0x83,0x3E,0xD4,0x71,0x6F,
        0x26,0x60,0x44,0xD4,0x9C,0x4B,0xDE,0x35,0x2E,0xB9,0x61,0x7A,0x2F,0x84,0xD7,0xDB,
        0x0A,0x39,0x21,0xFF,0xD7,0x64,0x2F,0x65,0x2C,0x0E,0x77,0x04,0x36,0x83,0x9F,0x2E,
        0x08,0x59,0x7D,0xBA,0x32,0xAD,0x42,0x62,0x96,0xB0,0xF2,0x6A,0x77,0x63,0x18,0x83,
        0xD3,0x9E,0xB7,0xEE,0xF3,0x57,0x83,0xE4,0x40,0x21,0x51,0x36,0x58,0xF3,0xCD,0x31
    };
    uint8_t* pt = NULL;
    pt = (uint8_t*)malloc(BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(pt + 128 * i, cpu_pt, 128 * sizeof(uint8_t));
    }
    //ct
    uint8_t cpu_ct[128] = { 0x00, };
    uint8_t* ct = NULL;
    ct = (uint8_t*)malloc(BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(ct + 128 * i, cpu_ct, 128 * sizeof(uint8_t));
    }
    //aad
    uint8_t cpu_aad[112] = {
        0xE7,0x60,0x76,0xA5,0xF7,0xD0,0x84,0x9C,0xE9,0xB4,0xEE,0x62,0xCD,0xAB,0x61,0x3E,
        0xDA,0xDF,0xF5,0x3A,0x05,0x42,0x5E,0x84,0xD3,0x17,0x66,0x14,0x6B,0xB5,0x9B,0x8F,
        0xEF,0x87,0x44,0x6B,0x0C,0x58,0x9A,0x55,0xD0,0xCD,0x35,0xCF,0xBC,0x33,0xC5,0x1E,
        0x1B,0x7B,0x02,0x63,0x92,0x83,0xE6,0x08,0xF7,0x98,0x19,0x5D,0x66,0x38,0xAD,0x83,
        0x92,0x1C,0x0B,0x8A,0x5A,0x76,0x2D,0xD6,0x9D,0x77,0x0E,0xC7,0x4E,0x14,0x5F,0x99,
        0xCB,0x0A,0x7C,0x88,0x6A,0x1F,0x9E,0x38,0x55,0xAB,0x52,0x55,0x99,0x58,0x2A,0x76,
        0x79,0x26,0xB2,0x48,0x08,0xB2,0xA5,0xE8,0xF7,0xB7,0x61,0x90,0xAB,0xD5,0x99,0x76
    };
    uint8_t* aad = NULL;
    aad = (uint8_t*)malloc(BlockSize * ThreadSize * 112 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(aad + 112 * i, cpu_aad, 112 * sizeof(uint8_t));
    }
    //mk
    uint8_t cpu_mk[16] = {
        0x43,0x60,0x77,0xD9,0xEF,0x6A,0x74,0xDC,0x3F,0xB2,0x37,0xFC,0xE6,0xEB,0x3D,0x11
    };
    uint8_t* mk = NULL;
    mk = (uint8_t*)malloc(BlockSize * ThreadSize * 16 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(mk + 16 * i, cpu_mk, 16 * sizeof(uint8_t));
    }
    //tag
    uint8_t cpu_tag[16] = {
        0x00,
    };
    uint8_t* tag = NULL;
    tag = (uint8_t*)malloc(BlockSize * ThreadSize * 16 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(tag + 16 * i, cpu_tag, 16 * sizeof(uint8_t));
    }

    int mk_len = 16; //klen
    int iv_len = 12; //nlen
    int pt_len = 16 * 8;	//plen
    int aad_len = 112;//alen
    int tag_len = 16;//tlen
    int ct_len = pt_len;
    //넘겨 줘야할 값들(g_tag, g_aad, g_ctr, g_pt, &g_key,g_H)
    //LEA_GCM_CTX* g_ctx = NULL;
    //cudaMalloc((void**)g_ctx, BlockSize * ThreadSize * sizeof(LEA_GCM_CTX));
    // pt
    uint8_t* g_pt = NULL;
    cudaMalloc((void**)&g_pt, BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    // ct
    uint8_t* g_ct = NULL;
    cudaMalloc((void**)&g_ct, BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    // aad //len값 마지막에 포함
    uint8_t* g_aad = NULL;
    cudaMalloc((void**)&g_aad, BlockSize * ThreadSize * 112 * sizeof(uint8_t));
    //tag 결과값 저장 
    uint8_t* g_tag = NULL;
    cudaMalloc((void**)&g_tag, BlockSize * ThreadSize * 16 * sizeof(uint8_t));
    //iv  
    uint8_t* g_iv = NULL;
    cudaMalloc((void**)&g_iv, BlockSize * ThreadSize * 12 * sizeof(uint8_t));
    //mk
    uint8_t* g_mk = NULL;
    cudaMalloc((void**)&g_mk, BlockSize * ThreadSize * 16 * sizeof(uint8_t));


    //cuda에 값 복제
    cudaMemcpy((void**)g_pt, pt, BlockSize * ThreadSize * 128 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_mk, mk, BlockSize * ThreadSize * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_tag, tag, BlockSize * ThreadSize * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_aad, aad, BlockSize * ThreadSize * 112 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_iv, iv, BlockSize * ThreadSize * 12 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_ct, ct, BlockSize * ThreadSize * 128 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // LEA_GCM_CTX* ctx = NULL;
    // ctx = (LEA_GCM_CTX*)malloc(BlockSize * ThreadSize * sizeof(LEA_GCM_CTX));
    // memset(ctx, 0, sizeof(LEA_GCM_CTX) * ThreadSize * BlockSize);
    // LEA_GCM_CTX* g_ctx = NULL;
    // cudaMalloc((void**)g_ctx, BlockSize * ThreadSize * sizeof(LEA_GCM_CTX));
    // cudaMemcpy((void**)g_ctx, ctx, BlockSize* ThreadSize * sizeof(LEA_GCM_CTX), cudaMemcpyHostToDevice);


     //성능측정
    cudaEvent_t start, stop;
    float elapsed_time_ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //Global
    GCM_REF << < BlockSize, ThreadSize >> > (g_mk, mk_len, g_iv, iv_len, g_aad, aad_len, g_ct, g_pt, pt_len, g_tag, tag_len);

    cudaDeviceSynchronize();
    //성능측정
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    printf("reference code\n");
    printf("elapsed_time_ms is %4.4f\n", elapsed_time_ms);
    printf("Performance : %4.2f GCM time per second \n", BlockSize * ThreadSize / ((elapsed_time_ms / 1000)));

    //cuda->cpu
    cudaMemcpy(tag, g_tag, BlockSize * 16 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(aad, g_aad, BlockSize * 112 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ct, g_ct, BlockSize * 128 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(pt, g_pt, BlockSize * 128 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(ctx, g_ctx, BlockSize  * ThreadSize * sizeof(LEA_GCM_CTX), cudaMemcpyDeviceToHost);



    cudaFree(g_pt);
    cudaFree(g_tag);
    cudaFree(g_aad);
    cudaFree(g_mk);
    cudaFree(g_ct);
    //cudaFree(g_ctx);
    cudaFree(g_iv);

    free(pt);
    //free(ctx);
    free(aad);
    free(mk);
    free(iv);
    free(tag);
    free(ct);
}
__global__ void GCM_4bit_table(uint8_t* mk, int mk_len, uint8_t* iv, int iv_len, uint8_t* aad, int aad_len, uint8_t* ct, uint8_t* pt, int pt_len, uint8_t* tag, int tag_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    LEA_GCM_CTX ctx;
    _lea_gcm_init_4(&ctx, mk + tid * 16, mk_len);
    gcm_set_ctr_4(&ctx, iv + tid * 12, iv_len);
    gcm_set_aad_4(&ctx, aad + tid * 112, aad_len);
    gcm_enc_4(&ctx, ct + tid * 128, pt + tid * 128, pt_len);
    gcm_final_4(&ctx, tag + tid * 16, tag_len);

}
void test_GPU_4bit_table(int BlockSize, int ThreadSize) {//이게 약간 main문 같은것 다 박으면 됨
    //ctx
   // LEA_GCM_CTX* ctx;
    //ctx = (LEA_GCM_CTX*)malloc(BlockSize * ThreadSize * sizeof(LEA_GCM_CTX));
    //iv
    uint8_t cpu_iv[12] = {
        0x67,0x61,0x32,0x6D,0xE2,0x80,0x63,0x78,0x2E,0x96,0x72,0x98
    };
    uint8_t* iv = NULL;
    iv = (uint8_t*)malloc(BlockSize * ThreadSize * 12 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(iv + 12 * i, cpu_iv, 12 * sizeof(uint8_t));
    }
    //pt
    uint8_t cpu_pt[128] = {
        0x29,0xDA,0x4D,0x54,0x4C,0xAC,0x60,0xB8,0x83,0x1E,0x0A,0xFB,0x3B,0xB4,0x4E,0x5F,
        0xBC,0x68,0xCC,0x59,0xB9,0xF1,0xEF,0xF2,0x25,0x45,0x67,0x6D,0x49,0x5D,0xDA,0x2A,
        0x14,0x38,0xD6,0xCA,0xB2,0x22,0x0B,0x94,0x60,0x36,0xB7,0x17,0x7E,0x22,0x61,0x11,
        0xE5,0x2A,0xCA,0x90,0x7C,0x70,0x21,0x57,0x06,0x72,0x76,0x83,0x3E,0xD4,0x71,0x6F,
        0x26,0x60,0x44,0xD4,0x9C,0x4B,0xDE,0x35,0x2E,0xB9,0x61,0x7A,0x2F,0x84,0xD7,0xDB,
        0x0A,0x39,0x21,0xFF,0xD7,0x64,0x2F,0x65,0x2C,0x0E,0x77,0x04,0x36,0x83,0x9F,0x2E,
        0x08,0x59,0x7D,0xBA,0x32,0xAD,0x42,0x62,0x96,0xB0,0xF2,0x6A,0x77,0x63,0x18,0x83,
        0xD3,0x9E,0xB7,0xEE,0xF3,0x57,0x83,0xE4,0x40,0x21,0x51,0x36,0x58,0xF3,0xCD,0x31
    };
    uint8_t* pt = NULL;
    pt = (uint8_t*)malloc(BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(pt + 128 * i, cpu_pt, 128 * sizeof(uint8_t));
    }
    //ct
    uint8_t cpu_ct[128] = { 0x00, };
    uint8_t* ct = NULL;
    ct = (uint8_t*)malloc(BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(ct + 128 * i, cpu_ct, 128 * sizeof(uint8_t));
    }
    //aad
    uint8_t cpu_aad[112] = {
        0xE7,0x60,0x76,0xA5,0xF7,0xD0,0x84,0x9C,0xE9,0xB4,0xEE,0x62,0xCD,0xAB,0x61,0x3E,
        0xDA,0xDF,0xF5,0x3A,0x05,0x42,0x5E,0x84,0xD3,0x17,0x66,0x14,0x6B,0xB5,0x9B,0x8F,
        0xEF,0x87,0x44,0x6B,0x0C,0x58,0x9A,0x55,0xD0,0xCD,0x35,0xCF,0xBC,0x33,0xC5,0x1E,
        0x1B,0x7B,0x02,0x63,0x92,0x83,0xE6,0x08,0xF7,0x98,0x19,0x5D,0x66,0x38,0xAD,0x83,
        0x92,0x1C,0x0B,0x8A,0x5A,0x76,0x2D,0xD6,0x9D,0x77,0x0E,0xC7,0x4E,0x14,0x5F,0x99,
        0xCB,0x0A,0x7C,0x88,0x6A,0x1F,0x9E,0x38,0x55,0xAB,0x52,0x55,0x99,0x58,0x2A,0x76,
        0x79,0x26,0xB2,0x48,0x08,0xB2,0xA5,0xE8,0xF7,0xB7,0x61,0x90,0xAB,0xD5,0x99,0x76
    };
    uint8_t* aad = NULL;
    aad = (uint8_t*)malloc(BlockSize * ThreadSize * 112 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(aad + 112 * i, cpu_aad, 112 * sizeof(uint8_t));
    }
    //mk
    uint8_t cpu_mk[16] = {
        0x43,0x60,0x77,0xD9,0xEF,0x6A,0x74,0xDC,0x3F,0xB2,0x37,0xFC,0xE6,0xEB,0x3D,0x11
    };
    uint8_t* mk = NULL;
    mk = (uint8_t*)malloc(BlockSize * ThreadSize * 16 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(mk + 16 * i, cpu_mk, 16 * sizeof(uint8_t));
    }
    //tag
    uint8_t cpu_tag[16] = {
        0x00,
    };
    uint8_t* tag = NULL;
    tag = (uint8_t*)malloc(BlockSize * ThreadSize * 16 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(tag + 16 * i, cpu_tag, 16 * sizeof(uint8_t));
    }

    int mk_len = 16; //klen
    int iv_len = 12; //nlen
    int pt_len = 16 * 8;	//plen
    int aad_len = 112;//alen
    int tag_len = 16;//tlen
    int ct_len = pt_len;
    //넘겨 줘야할 값들(g_tag, g_aad, g_ctr, g_pt, &g_key,g_H)
    //LEA_GCM_CTX* g_ctx = NULL;
    //cudaMalloc((void**)g_ctx, BlockSize * ThreadSize * sizeof(LEA_GCM_CTX));
    // pt
    uint8_t* g_pt = NULL;
    cudaMalloc((void**)&g_pt, BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    // ct
    uint8_t* g_ct = NULL;
    cudaMalloc((void**)&g_ct, BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    // aad //len값 마지막에 포함
    uint8_t* g_aad = NULL;
    cudaMalloc((void**)&g_aad, BlockSize * ThreadSize * 112 * sizeof(uint8_t));
    //tag 결과값 저장 
    uint8_t* g_tag = NULL;
    cudaMalloc((void**)&g_tag, BlockSize * ThreadSize * 16 * sizeof(uint8_t));
    //iv  
    uint8_t* g_iv = NULL;
    cudaMalloc((void**)&g_iv, BlockSize * ThreadSize * 12 * sizeof(uint8_t));
    //mk
    uint8_t* g_mk = NULL;
    cudaMalloc((void**)&g_mk, BlockSize * ThreadSize * 16 * sizeof(uint8_t));


    //cuda에 값 복제
    cudaMemcpy((void**)g_pt, pt, BlockSize * ThreadSize * 128 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_mk, mk, BlockSize * ThreadSize * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_tag, tag, BlockSize * ThreadSize * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_aad, aad, BlockSize * ThreadSize * 112 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_iv, iv, BlockSize * ThreadSize * 12 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_ct, ct, BlockSize * ThreadSize * 128 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    //cudaMemcpy((void**)g_ctx, ctx, BlockSize* ThreadSize * sizeof(LEA_GCM_CTX), cudaMemcpyHostToDevice);


    //성능측정
    cudaEvent_t start, stop;
    float elapsed_time_ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //Global
    GCM_4bit_table << < BlockSize, ThreadSize >> > (g_mk, mk_len, g_iv, iv_len, g_aad, aad_len, g_ct, g_pt, pt_len, g_tag, tag_len);

    cudaDeviceSynchronize();
    //성능측정
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    printf("4bit table version code\n");
    printf("elapsed_time_ms is %4.4f\n", elapsed_time_ms);
    printf("Performance : %4.2f GCM time per second \n", BlockSize * ThreadSize / ((elapsed_time_ms / 1000)));

    //cuda->cpu
    cudaMemcpy(tag, g_tag, BlockSize * 16 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(aad, g_aad, BlockSize * 112 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ct, g_ct, BlockSize * 128 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(pt, g_pt, BlockSize * 128 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(ctx, g_ctx, BlockSize  * ThreadSize * sizeof(LEA_GCM_CTX), cudaMemcpyDeviceToHost);



    cudaFree(g_pt);
    cudaFree(g_tag);
    cudaFree(g_aad);
    cudaFree(g_mk);
    cudaFree(g_ct);
    //cudaFree(g_ctx);
    cudaFree(g_iv);

    free(pt);
    //free(ctx);
    free(aad);
    free(mk);
    free(iv);
    free(tag);
    free(ct);
}
__global__ void GCM_8bit_table(uint8_t* mk, int mk_len, uint8_t* iv, int iv_len, uint8_t* aad, int aad_len, uint8_t* ct, uint8_t* pt, int pt_len, uint8_t* tag, int tag_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    LEA_GCM_CTX ctx;
    _lea_gcm_init_8(&ctx, mk + tid * 16, mk_len);
    gcm_set_ctr_8(&ctx, iv + tid * 12, iv_len);
    gcm_set_aad_8(&ctx, aad + tid * 112, aad_len);
    gcm_enc_8(&ctx, ct + tid * 128, pt + tid * 128, pt_len);
    gcm_final_8(&ctx, tag + tid * 16, tag_len);
}
void test_GPU_8bit_table(int BlockSize, int ThreadSize) {//이게 약간 main문 같은것 다 박으면 됨
    //ctx
   // LEA_GCM_CTX* ctx;
    //ctx = (LEA_GCM_CTX*)malloc(BlockSize * ThreadSize * sizeof(LEA_GCM_CTX));
    //iv
    uint8_t cpu_iv[12] = {
        0x67,0x61,0x32,0x6D,0xE2,0x80,0x63,0x78,0x2E,0x96,0x72,0x98
    };
    uint8_t* iv = NULL;
    iv = (uint8_t*)malloc(BlockSize * ThreadSize * 12 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(iv + 12 * i, cpu_iv, 12 * sizeof(uint8_t));
    }
    //pt
    uint8_t cpu_pt[128] = {
        0x29,0xDA,0x4D,0x54,0x4C,0xAC,0x60,0xB8,0x83,0x1E,0x0A,0xFB,0x3B,0xB4,0x4E,0x5F,
        0xBC,0x68,0xCC,0x59,0xB9,0xF1,0xEF,0xF2,0x25,0x45,0x67,0x6D,0x49,0x5D,0xDA,0x2A,
        0x14,0x38,0xD6,0xCA,0xB2,0x22,0x0B,0x94,0x60,0x36,0xB7,0x17,0x7E,0x22,0x61,0x11,
        0xE5,0x2A,0xCA,0x90,0x7C,0x70,0x21,0x57,0x06,0x72,0x76,0x83,0x3E,0xD4,0x71,0x6F,
        0x26,0x60,0x44,0xD4,0x9C,0x4B,0xDE,0x35,0x2E,0xB9,0x61,0x7A,0x2F,0x84,0xD7,0xDB,
        0x0A,0x39,0x21,0xFF,0xD7,0x64,0x2F,0x65,0x2C,0x0E,0x77,0x04,0x36,0x83,0x9F,0x2E,
        0x08,0x59,0x7D,0xBA,0x32,0xAD,0x42,0x62,0x96,0xB0,0xF2,0x6A,0x77,0x63,0x18,0x83,
        0xD3,0x9E,0xB7,0xEE,0xF3,0x57,0x83,0xE4,0x40,0x21,0x51,0x36,0x58,0xF3,0xCD,0x31
    };
    uint8_t* pt = NULL;
    pt = (uint8_t*)malloc(BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(pt + 128 * i, cpu_pt, 128 * sizeof(uint8_t));
    }
    //ct
    uint8_t cpu_ct[128] = { 0x00, };
    uint8_t* ct = NULL;
    ct = (uint8_t*)malloc(BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(ct + 128 * i, cpu_ct, 128 * sizeof(uint8_t));
    }
    //aad
    uint8_t cpu_aad[112] = {
        0xE7,0x60,0x76,0xA5,0xF7,0xD0,0x84,0x9C,0xE9,0xB4,0xEE,0x62,0xCD,0xAB,0x61,0x3E,
        0xDA,0xDF,0xF5,0x3A,0x05,0x42,0x5E,0x84,0xD3,0x17,0x66,0x14,0x6B,0xB5,0x9B,0x8F,
        0xEF,0x87,0x44,0x6B,0x0C,0x58,0x9A,0x55,0xD0,0xCD,0x35,0xCF,0xBC,0x33,0xC5,0x1E,
        0x1B,0x7B,0x02,0x63,0x92,0x83,0xE6,0x08,0xF7,0x98,0x19,0x5D,0x66,0x38,0xAD,0x83,
        0x92,0x1C,0x0B,0x8A,0x5A,0x76,0x2D,0xD6,0x9D,0x77,0x0E,0xC7,0x4E,0x14,0x5F,0x99,
        0xCB,0x0A,0x7C,0x88,0x6A,0x1F,0x9E,0x38,0x55,0xAB,0x52,0x55,0x99,0x58,0x2A,0x76,
        0x79,0x26,0xB2,0x48,0x08,0xB2,0xA5,0xE8,0xF7,0xB7,0x61,0x90,0xAB,0xD5,0x99,0x76
    };
    uint8_t* aad = NULL;
    aad = (uint8_t*)malloc(BlockSize * ThreadSize * 112 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(aad + 112 * i, cpu_aad, 112 * sizeof(uint8_t));
    }
    //mk
    uint8_t cpu_mk[16] = {
        0x43,0x60,0x77,0xD9,0xEF,0x6A,0x74,0xDC,0x3F,0xB2,0x37,0xFC,0xE6,0xEB,0x3D,0x11
    };
    uint8_t* mk = NULL;
    mk = (uint8_t*)malloc(BlockSize * ThreadSize * 16 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(mk + 16 * i, cpu_mk, 16 * sizeof(uint8_t));
    }
    //tag
    uint8_t cpu_tag[16] = {
        0x00,
    };
    uint8_t* tag = NULL;
    tag = (uint8_t*)malloc(BlockSize * ThreadSize * 16 * sizeof(uint8_t));
    for (int i = 0; i < BlockSize * ThreadSize; i++) {
        memcpy(tag + 16 * i, cpu_tag, 16 * sizeof(uint8_t));
    }

    int mk_len = 16; //klen
    int iv_len = 12; //nlen
    int pt_len = 16 * 8;	//plen
    int aad_len = 112;//alen
    int tag_len = 16;//tlen
    int ct_len = pt_len;
    //넘겨 줘야할 값들(g_tag, g_aad, g_ctr, g_pt, &g_key,g_H)
    //LEA_GCM_CTX* g_ctx = NULL;
    //cudaMalloc((void**)g_ctx, BlockSize * ThreadSize * sizeof(LEA_GCM_CTX));
    // pt
    uint8_t* g_pt = NULL;
    cudaMalloc((void**)&g_pt, BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    // ct
    uint8_t* g_ct = NULL;
    cudaMalloc((void**)&g_ct, BlockSize * ThreadSize * 128 * sizeof(uint8_t));
    // aad //len값 마지막에 포함
    uint8_t* g_aad = NULL;
    cudaMalloc((void**)&g_aad, BlockSize * ThreadSize * 112 * sizeof(uint8_t));
    //tag 결과값 저장 
    uint8_t* g_tag = NULL;
    cudaMalloc((void**)&g_tag, BlockSize * ThreadSize * 16 * sizeof(uint8_t));
    //iv  
    uint8_t* g_iv = NULL;
    cudaMalloc((void**)&g_iv, BlockSize * ThreadSize * 12 * sizeof(uint8_t));
    //mk
    uint8_t* g_mk = NULL;
    cudaMalloc((void**)&g_mk, BlockSize * ThreadSize * 16 * sizeof(uint8_t));


    //cuda에 값 복제
    cudaMemcpy((void**)g_pt, pt, BlockSize * ThreadSize * 128 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_mk, mk, BlockSize * ThreadSize * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_tag, tag, BlockSize * ThreadSize * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_aad, aad, BlockSize * ThreadSize * 112 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_iv, iv, BlockSize * ThreadSize * 12 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy((void**)g_ct, ct, BlockSize * ThreadSize * 128 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    //cudaMemcpy((void**)g_ctx, ctx, BlockSize* ThreadSize * sizeof(LEA_GCM_CTX), cudaMemcpyHostToDevice);


    //성능측정
    cudaEvent_t start, stop;
    float elapsed_time_ms = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //Global
    GCM_8bit_table << < BlockSize, ThreadSize >> > (g_mk, mk_len, g_iv, iv_len, g_aad, aad_len, g_ct, g_pt, pt_len, g_tag, tag_len);

    cudaDeviceSynchronize();
    //성능측정
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    printf("8bit table version code\n");
    printf("elapsed_time_ms is %4.4f\n", elapsed_time_ms);
    printf("Performance : %4.2f GCM time per second \n", BlockSize * ThreadSize / ((elapsed_time_ms / 1000)));

    //cuda->cpu
    cudaMemcpy(tag, g_tag, BlockSize * 16 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(aad, g_aad, BlockSize * 112 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ct, g_ct, BlockSize * 128 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(pt, g_pt, BlockSize * 128 * ThreadSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(ctx, g_ctx, BlockSize  * ThreadSize * sizeof(LEA_GCM_CTX), cudaMemcpyDeviceToHost);



    cudaFree(g_pt);
    cudaFree(g_tag);
    cudaFree(g_aad);
    cudaFree(g_mk);
    cudaFree(g_ct);
    //cudaFree(g_ctx);
    cudaFree(g_iv);

    free(pt);
    //free(ctx);
    free(aad);
    free(mk);
    free(iv);
    free(tag);
    free(ct);
}

int main() {
    test_GPU_parallel(64, 32);
    //printf("\n\n");
    test_GPU_REF(16, 32);
    //printf("\n\n");
    test_GPU_4bit_table(16, 32);
    //printf("\n\n");
    test_GPU_8bit_table(16, 32);
}


