//--- function h related (h is called lambda in the paper)
#define h_exp exp                   // ideal
#define h_log log                   // ideal
#define h_tanh2(a)  (tanh(a/2.0))   // ideal
#define h_atanh2(x) (atanh(x)*2)    // ideal
//-- fast lib: by tables or numerical methods to speed up
/*#define h_exp exp_fast
#define h_log log_fast
#define h_tanh2(a)  ((h_exp(a)-1.0)/(h_exp(a)+1.0))
#define h_atanh2(x) (h_log((1.0+x)/(1.0-x)))*/


#define h_xyz(LA,type) ( \
  ((type)==0? ( h_log( ( 1.0+h_exp(-LA[0]) ) / (h_exp(-LA[1])+h_exp(-LA[2])) ) ) : \
  ((type)==1? ( h_log( ( 1.0+h_exp(-LA[1]) ) / (h_exp(-LA[0])+h_exp(-LA[2])) ) ) : \
              ( h_log( ( 1.0+h_exp(-LA[2]) ) / (h_exp(-LA[0])+h_exp(-LA[1])) ) ) ) \
  ) \
)

#define bplus(a,b)  (  h_atanh2( h_tanh2(a) * h_tanh2(b) )  )
#define bminus(a,b) (  h_atanh2( h_tanh2(a) / h_tanh2(b) )  )
static inline double bminus_protected(double a, double b)
{
  double x = h_tanh2(a) / h_tanh2(b);
  if(x>0.999999999999999)       x = 0.999999999999999;    //atanh(0.999999999999999)*2 = 35.2327231729 ~= LLR_MAX
  else if(x<-0.999999999999999) x =-0.999999999999999;
  return( h_atanh2(x) );
}


//-- restrict LLR in reasonable range
#define LLR_MAX (35)  // IEEE754 double tanh(35.9/2) = 0.999999999999999 , but tanh(36/2) = 1
#define LLR_MIN (1.0e-10)  // support up to row weight 30, since tanh(1e-10/2)^30 = 9.313225746154796e-310 ~= |DBL_MIN|
#define restrict_llr(llr) { \
  if(llr > LLR_MAX) { llr = LLR_MAX; } \
  else if(llr < LLR_MIN) { \
    if(llr >= 0)          { llr =  LLR_MIN; } \
    else if(llr>-LLR_MIN) { llr = -LLR_MIN; } \
    else if(llr<-LLR_MAX) { llr = -LLR_MAX; } \
  } \
}



//-- Serial LLR-based MBP4
int32_t Lbp_dec64(LBP_Ctl *bp, a_matrix_GFQ *A)
{
  int32_t m, n, i, j, q;      double Dmj;       double x;

 if(bp->reset == 1) { //======== make Dm ready first ========//
  for(m=0; m < A->MM; m++) {
    bp->Dm[m] = (bp->target_z[m]==0)? bp->Lmj[m][0] : -bp->Lmj[m][0];   // ASSUME no zero-weight row; also handle z_m here
    for(j=1; j < A->num_m[m]; j++) {   // NOTE !! start from j=1
      bp->Dm[m] = bplus(bp->Dm[m], bp->Lmj[m][j]);
    } // (Dm ready after this loop)
  }
 }  //======== make Dm ready done ========//


  memcpy ( bp->GA , bp->LA, sizeof(bp->GA) );  // init GAMMA as LAMBDA

  //======== main process of this iteration ========//
  for(n=0; n < A->NN; n++) 
  {
    //-------- Combine from m \in \sM(n) --------//
    for(i=0; i < A->num_n[n]; i++) {
      m = A->nlist[n][i];
      j = A->ni2j[n][i];

      bp->Dmj[m][j] = bminus_protected(bp->Dm[m], bp->Lmj[m][j]);  // serial schedule needs this BECAUSE |Dm[m]| may > |Lmj[m][j]|

      Dmj = bp->Dmj[m][j] ;   // shorthand notation

      //-- apply ALPHA
      Dmj *= ALPHA;

      q = A->mlist_val[m][j]; // {X,Z,Y} val {1,2,3}
      switch (q) {
      case 1: bp->GA[n][1]+=Dmj;  bp->GA[n][2]+=Dmj;  break;  // edge is X: q=1, type=0
      case 2: bp->GA[n][0]+=Dmj;  bp->GA[n][2]+=Dmj;  break;  // edge is Z: q=2, type=1
      case 3: bp->GA[n][0]+=Dmj;  bp->GA[n][1]+=Dmj;  break;  // edge is Y: q=3, type=2
      }
    }

    //-------- Dispatch to m \in \sM(n) --------//
    for(i=0; i < A->num_n[n]; i++) {
      m = A->nlist[n][i];
      j = A->ni2j[n][i];
      q = A->mlist_val[m][j]; // {X,Z,Y} val {1,2,3}

      //- update Lmj // recall that // q type = q-1
      bp->Lmj[m][j] = h_xyz(bp->GA[n], q-1) - bp->Dmj[m][j];

      #if 1 // LLR MAX and LLR_MIN
      restrict_llr(bp->Lmj[m][j]);
      #endif

      //- make Dm more updated (KEY_STEP for Serial BP along variable nodes)
      bp->Dm[m] = bplus(bp->Dmj_ori[m][j], bp->Lmj[m][j]);
    }
  } //======== END of main process of this iteration ========//

  bp->reset = 0;
  bp->iter += 1;
  //======== hard decision ========//
  HardDecisionLLR(bp, A);
  //======== Syndrome Checking ========//
  Quan_GenSyndrome(A, bp->tt, bp->zz);

  //-- check if meet target_z 
  if( 0 == HamDist(bp->zz, bp->target_z, A->MM) )  return 1;
  else                                             return 0;
}
