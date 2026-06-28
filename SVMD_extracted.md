# Successive variational mode decomposition

Mojtaba Nazari, Sayed Mahmoud Sakhaei∗

<!-- image-->

Department of computer and electrical engineering, Babol Noshirvani University of Technology, Babol, Iran

a r t i c l e i n f o

Article history:   
Received 1 November 2019   
Revised 11 February 2020   
Accepted 12 April 2020   
Available online 5 May 2020

Keywords:   
Variational mode decomposition   
Compact spectrum   
Alternate direction method of multipliers   
algorithm (ADMM)

## a b s t r a c t

Variational mode decomposition (VMD) is a powerful technique for concurrently decomposing a signal into its constituent intrinsic modes. However, the performance of VMD will be degraded if the number of modes available in the signal is not precisely known. In this paper, we introduce a new method, namely successive variational mode decomposition (SVMD), which extracts the modes successively and does not need to know the number of modes. The method considers the mode as a signal with maximally compact spectrum, as VMD does. It achieves the mode decomposition by adding some criteria to the optimization problem of VMD: the mode of interest has no or less spectral overlap to the other modes and to the residual signal. Our simulations on some artificial and real world data have demonstrated that the new method without knowing the number of modes converges to the same modes as VMD does with knowing the precise number of modes. Moreover, the computational complexity of SVMD is much lower than that of VMD. Another advantage of SVMD over VMD is more robustness against the initial values of the center frequencies of modes.

© 2020 Elsevier B.V. All rights reserved.

## 1. Introduction

Empirical mode decomposition (EMD) is the most well-known method for decomposing a signal into a set of zero-mean and almost the same number of maxima and minima components [1]. It recursively decomposes a nonstationary signal into a datadependent basis functions termed Intrinsic Mode Function (IMF). The important feature of EMD is that it is capable to represent a nonlinear and nonstationary signal as a sum of physically meaningful time-frequency components. EMD was satisfactorily used in many applications such as extracting the vital signs from biological signals [2–5], removing interferences from signal [6,7], and climate analysis [8,9]. However, the results of EMD are highly affected by the methods of finding and interpolating extremal points [10,11]. Therefore, some EMD-like methods were developed in order to cope with the low robustness against noise and the lack of mathematical theory [12–14].

Unlike EMD, Variational Mode Decomposition (VMD) method is fully established on a mathematical framework [15]. VMD considers the modes as narrowband signals with separate bands squeezed around different center frequencies. In comparison to EMD based methods, VMD is more robust against noise and sampling error. This advantage makes VMD a useful tool in many applications such as detecting rub-impact fault of the rotor system and bearing fault diagnosis [16,17], signal denoising [18–20], speech signal processing [21], and seismic time-frequency analysis [22,23]. Moreover, whereas EMD extracts the modes recursively, VMD does it concurrently through iterative procedure. In the mathematical implementation view, VMD looks for K modes as the signals which simultaneously minimize a proper optimization problem. The complexity of the optimization problem increases by increasing K, resulting in a slow convergence of the algorithm.

One of the main problems of VMD is the proper setting K before the algorithm runs. High value of K may result in mode mixing or purely noisy modes, whereas low K may lead to duplicate modes. Some researchers developed a method based on the detrended fluctuation analysis (DFA), for determining K adaptively [19,24]. This method determines K by evaluating the scaling exponent from the noisy signal and is considered as a preprocessing step. Therefore, it has an extra computational burden besides VMD.

Other variants of VMD were also introduced for processing 2D real signals (images) named as 2D-VMD [25] and for complexvalued signals named as Complex-VMD [26], both of which has the same structure as VMD. Moreover, 2D-VMD for complex valued signals has been presented in [27] for denoising purpose of the seismic data. All of these variants of VMD are based on the assumption that the modes are narrowband. An extension of VMD to decompose a signal containing wideband nonlinear chirp components was given in [28]. The method named as variational nonlinear chirp mode decomposition (VNCMD) analyses the signal by transforming wideband nonlinear chirp components to a narrowband signal through demodulation techniques.

Recently, we introduced a new method for extracting a specific mode of a signal and its application for ECG-Derived-Respiration (EDR) signal [29]. The method termed Variational Mode Extraction (VME) extracts an intrinsic mode function by knowing its approximate center frequency. This method is established on the same concepts utilized in VMD but, VME extracts the mode of interest independent from other modes. In this paper, we extend the VME and propose an efficient and fast adaptive method for variational decomposition of a signal. This new decomposition method extracts all IMFs in a successive manner, in contrary to VMD where the modes are concurrently extracted. This successive approach leads to a method with no need to know the number of modes and lower computational complexity in comparison to VMD.

The paper is organized as follows. In section 2, we explain our new decomposition method named as Successive-VMD (SVMD) method in details. Section 3 shows some experiments based on simulated and real signals and contains the comparisons of results obtained by VMD. Finally, we conclude and give some discussions, perspectives and remarks in Section 4.

## 2. Decomposition of signal using Successive-VMD

As explained before, VMD method finds IMFs simultaneously. Therefore, the computational time would dramatically be increased when the number of modes, K, available in the signal is high. Moreover, there are applications where some modes are considered as interference or noise signal and the corresponding modes are not in interest. SVMD is an algorithm which finds modes one after the other and this succession helps increase convergence rate and also not extract the unwanted modes, both decrease the computational time.

## 2.1. The new method: SVMD

Decomposition is here done by successively applying VME on the signal, where some constraints are added to avoid converging to the previously extracted modes. This procedure is continued until all modes are extracted or the reconstruction error (the error between the input signal and sum of the modes) is less than a threshold. In other words, assume L-1 modes are found and it is desired to determine the next mode. To that end, an optimization problem is solved to find a signal with maximally compact spectrum (i.e., the Lth mode) when added to the sum of the extracted modes reduces the reconstruction error.

To mathematically represent the method, we assume that the input signal f(t) is decomposed into two signals: the Lth mode (uL(t)) and the residual signal (fr(t)) as follows:

$$
\begin{array} { r } { f ( t ) = u _ { L } ( t ) + f _ { r } ( t ) . } \end{array}\tag{1}
$$

where, the residual signal $f _ { r } ( t )$ is the input signal other than uL(t) and contains two parts: the sum of the previously obtained modes $\begin{array} { r } { \big ( \sum _ { i = 1 : L - 1 } u _ { i } ( t ) \big ) } \end{array}$ and the un-processed part of signal $( f _ { u } ( t ) )$ :

$$
f _ { r } ( t ) = \sum _ { i = 1 : L - 1 } u _ { i } ( t ) + f _ { u } ( t ) .\tag{2}
$$

It is clear that for finding the first mode, the first part of $f _ { r } ( t )$ (sum of the previously obtained modes) is zero. Now, we represent our proposed decomposition method in details based on four criteria, three of which are the same as VME:

(1) Each mode should be compact around its center frequency. Consequently, the Lth mode minimizes the following criterion:

$$
J _ { 1 } = \left. \partial _ { t } \bigg [ \bigg ( \delta ( t ) + \frac { j } { \pi t } \bigg ) * u _ { L } ( t ) \bigg ] e ^ { - j \omega _ { L } t } \right. _ { 2 } ^ { 2 } .\tag{3}
$$

where $\omega _ { L }$ is the center frequency of the Lth mode and ∗ denotes the convolution operation.

(2) The energy of the residual signal $f _ { r } ( t )$ should be minimized at frequencies where $u _ { L } ( t )$ has effective components [15]. As suggested in VME method [29], this constraint is realized by using a proper filter $\hat { \beta } _ { L } ( \omega )$ with frequency response of:

$$
\hat { \beta } _ { L } ( \omega ) = \frac { 1 } { \alpha \left( \omega - \omega _ { L } \right) ^ { 2 } } .\tag{4}
$$

Therefore, to get minimized spectral overlap between $f _ { r } ( t )$ and $u _ { L } ( t )$ , the energy of filtered $f _ { r } ( t )$ by $\hat { \beta } _ { L } ( \omega )$ should be minimized. Consequently, we consider the following criterion to minimize the spectral overlap of residual signal and the Lth mode:

$$
J _ { 2 } = \| \beta _ { \mathrm { L } } ( t ) * f _ { r } ( t ) \| _ { 2 } ^ { 2 } .\tag{5}
$$

where $\beta _ { L } ( t )$ is the impulse response of the filter described by (4).

(3) By minimizing the two criteria $J _ { 1 }$ and $J _ { 2 } ,$ the Lth mode may be obtained as one of the previously L-1 modes. To avoid this situation, it is helpful to notice that $u _ { L } ( t )$ should have less energy at frequencies around the center frequencies of the previously obtained modes. This constraint can be satisfied by the similar approach used in establishing the criterion $J _ { 2 } ,$ , which is using proper filters with the frequency responses as:

$$
\hat { \beta } _ { i } ( \omega ) = \frac { 1 } { \alpha { ( \omega - \omega _ { i } ) } ^ { 2 } } ; i = 1 , \ 2 , \ \ldots , \ L - 1 ,\tag{6}
$$

Therefore, the so called added criterion is represented as follows:

$$
J _ { 3 } = \sum _ { i = 1 } ^ { L - 1 } \left. \beta _ { i } ( t ) \ast u _ { L } ( t ) \right. _ { 2 } ^ { 2 } ,\tag{7}
$$

where $\beta _ { i } ( t )$ is the impulse response of the filter in (6).

(4) The last constraint is to guarantee complete reconstruction of $f ( t )$ from L modes and the un-processed part of the signal:

$$
f ( t ) = u _ { L } ( t ) + f _ { u } ( t ) + \sum _ { i = 1 : L - 1 } u _ { i } ( t ) ,\tag{8}
$$

Hence, when L-1 modes are known, the problem of extracting the Lth mode can be expressed as a constrained minimization problem, in which a combination of $J _ { 1 } , J _ { 2 }$ and $J _ { 3 }$ is minimized subject to the constraint of (8):

$$
\begin{array} { r l } & { \underset { u _ { L } . \ \omega _ { L } . \ f _ { r } } { \operatorname* { m i n } } \left\{ \alpha J _ { 1 } + J _ { 2 } + J _ { 3 } \right\} } \\ & { s u b j e c t t o : u _ { L } ( t ) + f _ { r } ( t ) = f ( t ) . } \end{array}\tag{9}
$$

where $\alpha$ is a parameter for balancing $J _ { 1 } , J _ { 2 }$ and $J _ { 3 } ,$ which can be solved through Lagrangian multiplier method. It is noteworthy that we considered just one weighting parameter for simplifying the problem. Besides, this form of combination leads to the mode decomposition similar to that of VMD, as we will show in the simulations and also mathematically verify in Appendix A. For the sake of better convergence behavior and to encourage reconstruction fidelity in the presence of noise, we considered a combination of quadratic penalty term and Lagrangian multiplier to establish the augmented Lagrangian function, as follows:

$$
\begin{array} { r l } & { \mathcal { L } ( u _ { L } . \omega _ { L } . \lambda ) : = \alpha J _ { 1 } + J _ { 2 } + J _ { 3 } + \Bigg \| f ( t ) - \Bigg ( u _ { L } ( t ) + f _ { u } ( t ) + \sum _ { i = 1 } ^ { L - 1 } u _ { i } ( t ) \Bigg ) \Bigg \| _ { 2 } ^ { 2 } } \\ & { \quad \quad + \Bigg \langle \lambda ( t ) , f ( t ) - \Bigg ( u _ { L } ( t ) + f _ { u } ( t ) + \sum _ { i = 1 } ^ { L - 1 } u _ { i } ( t ) \Bigg ) \Bigg \rangle } \end{array}\tag{10}
$$

where λ refers to Lagrangian multiplier. After using Parseval’s equality and then by change of variables $\omega  \omega - \omega _ { L }$ in the first term, (13) can be rewritten as:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } ( \bar { \boldsymbol { \nu } } _ { \mathcal { L } } , \omega _ { \mathcal { L } } , \lambda ) = \alpha \ | | ( \omega - \omega _ { \mathcal { L } } ) | ( \cos + s g n ( \omega ) ) \hat { \boldsymbol { \nu } } _ { \mathrm { L } } ( \omega ) } | | _ { \epsilon } ^ { 2 } } } \\ { { \displaystyle ~ + \| \ \hat { \boldsymbol { \beta } } _ { l } ( \omega ) \bigg ( \hat { f } _ { a } ( \omega ) + \sum _ { i = 1 } ^ { L - 1 } \bar { u } _ { i } ( \omega ) \bigg ) \| _ { 2 } ^ { 2 } + \sum _ { i = 1 } ^ { L - 1 } \| \ \hat { \beta } _ { i } ( \omega ) , u _ { \boldsymbol { L } } ( \omega ) \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle ~ + \| \ \hat { f } _ { ( \omega ) } - \bigg ( \bar { u } _ { l } ( \omega ) + \hat { f } _ { n } ( \omega ) + \sum _ { k = 1 } ^ { L - 1 } \bar { u } _ { i } ( \omega ) \bigg ) \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle ~ + \{ \hat { \lambda } ( \omega ) , \hat { f } ( \omega ) - \bigg ( \bar { u } _ { \boldsymbol { L } } ( \omega ) + \hat { f } _ { n } ( \omega ) + \sum _ { i = 1 } ^ { L - 1 } \bar { u } _ { i } ( \omega ) \bigg ) } \| _ { 2 } ^ { 2 } }  \end{array}\tag{. (11}
$$

Now, as in VMD and VME methods, we use the alternate direction method of multipliers algorithm (ADMM) [15,30] to solve iteratively this minimization problem. This technique results in the following equation to update uL in the (n+1)th iteration:

Eqs. (14)–(16) represent one iteration for updating the optimization variables $\hat { u } _ { L } ( \omega ) , \omega _ { L }$ and $\hat { f } _ { u } ( \omega )$ . It is seen that the updated value of $\hat { f } _ { u } ( \omega )$ is not dependent in its current value. It therefore obviates the need for updating $\hat { f } _ { u } ( \omega )$ , if we replace $\hat { f } _ { u } ^ { n } \left( \omega \right)$ in (14) and (15) by its value calculated by (16). The result for $\hat { u } _ { L } ( \omega )$ is:

$$
\begin{array} { r l } & { \hat { u } _ { L } ^ { n + 1 } ( \omega ) } \\ & { = \frac { \hat { f } ( \omega ) + \alpha ^ { 2 } \big ( \omega - \omega _ { L } ^ { n } \big ) ^ { 4 } \hat { u } _ { L } ^ { n } ( \omega ) + \frac { \hat { \lambda } ( \omega ) } { 2 } } { \bigg [ 1 + \alpha ^ { 2 } \big ( \omega - \omega _ { L } ^ { n } \big ) ^ { 4 } \bigg ] \bigg [ 1 + 2 \alpha \big ( \omega - \omega _ { L } ^ { n } \big ) ^ { 2 } + \sum _ { i = 1 } ^ { L - 1 } \frac { 1 } { \alpha ^ { 2 } \big ( \omega - \omega _ { i } \big ) ^ { 4 } } \bigg ] } , } \end{array}\tag{17}
$$

Moreover, the second term at the numerator of (15) can be ignored against the first term, as α is generally very big, as explained in VME paper [29]. Therefore, the equation for updating $\omega _ { L }$ can approximately be written as follows:

$$
\omega _ { L } ^ { n + 1 } = \frac { \int _ { 0 } ^ { \infty } \omega \big | \hat { u } _ { L } ^ { n + 1 } ( \omega ) \big | ^ { 2 } d \omega } { \int _ { 0 } ^ { \infty } \big | \hat { u } _ { L } ^ { n + 1 } ( \omega ) \big | ^ { 2 } d \omega } .\tag{18}
$$

$$
\hat { u } _ { L } ^ { n + 1 } \gets \begin{array} { l } { \mathrm { a r g ~ m i n } } \\ { u _ { L } \in \mathcal { X } } \end{array} \left\{ \begin{array} { l } { \alpha \parallel j ( \omega - \omega _ { L } ) \big [ ( 1 + s g n ( \omega ) ) \hat { u } _ { L } ( \omega ) \big ] \| _ { 2 } ^ { 2 } + \parallel \hat { \beta } _ { L } ( \omega ) \left( \hat { f } _ { u } ( \omega ) + \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } ( \omega ) \right) \| _ { 2 } ^ { 2 } } \\ { + \displaystyle \sum _ { i = 1 } ^ { L - 1 } \parallel \hat { \beta } _ { i } ( \omega ) . u _ { L } ( \omega ) \| _ { 2 } ^ { 2 } + \parallel \hat { f } ( \omega ) - \left( \hat { u } _ { L } ( \omega ) + \hat { f } _ { u } ( \omega ) + \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } ( \omega ) \right) + \frac { \hat { \lambda } ( \omega ) } { 2 } \| _ { 2 } ^ { 2 } } \end{array} \right\} \mathbb { 1 } 2 )
$$

By representing all terms as half-space integrals over the nonnegative frequencies and incorporating the used filters i.e. $\hat { \beta } _ { i } ( \omega ) =$ $1 / \alpha ( \omega - \omega _ { i } ^ { n } ) ^ { 2 }$ and $\hat { \beta } _ { L } ( \omega ) = 1 / \alpha ( \omega - \omega _ { L } ^ { n } ) ^ { 2 } , ( 1 2 )$ can be rewritten as follows:

which is the same as one obtained by VME. Finally, the equation for updating the Lagrangian multiplier λ is achieved through dual ascent method [30]:

$$
\hat { u } _ { L } ^ { n + 1 } \gets \displaystyle { u _ { L } \varphi _ { \mathbf { \Phi } } } \displaystyle  m i n \left\{ \begin{array} { l l } { 4 \alpha \int _ { 0 } ^ { \infty } \left( \omega - \omega _ { L } \right) ^ { 2 } \left| \hat { u } _ { L } ( \omega ) \right| ^ { 2 } d \omega } \\ { + 2 \displaystyle { \sum _ { i = 1 } ^ { \infty } \int _ { 0 } ^ { \infty } \left| \frac { 1 } { \alpha \left( \omega - \omega _ { i } \right) ^ { 2 } } \hat { u } _ { L } ( \omega ) \right| ^ { 2 } d \omega } } \\ { + 2 \displaystyle { \int _ { 0 } ^ { \infty } \left| \frac { 1 } { \alpha \left( \omega - \omega _ { L } \right) ^ { 2 } } \left( \hat { f } _ { u } ( \omega ) + \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } ( \omega ) \right) \right| ^ { 2 } d \omega } } \\ { + 2 \displaystyle { \int _ { 0 } ^ { \infty } \left| \hat { f } ( \omega ) - \left( \hat { u } _ { L } ( \omega ) + \hat { f } _ { u } ( \omega ) + \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } ( \omega ) \right) + \frac { \hat { \mathbf { \Phi } } _ { \bot } ( \omega ) } { 2 } \right| ^ { 2 } d \omega } } \end{array} \right\} .\tag{13}
$$

Now, by letting the first variation vanish for the positive frequencies i.e. by minimizing (13) with respect to $\left( w . r . t \right) \ \hat { u } _ { L } ( \omega )$ , we would have:

$$
\hat { u } _ { L } ^ { n + 1 } ( \omega ) = \frac { \hat { f } ( \omega ) - \left( \hat { f } _ { u } ^ { n } ~ ( \omega ) + \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } ~ ( \omega ) \right) + \frac { \hat { \lambda } ( \omega ) } { 2 } } { 1 + 2 \alpha \left( \omega - \omega _ { L } ^ { n } \right) ^ { 2 } + \sum _ { i = 1 } ^ { L - 1 } \frac { 1 } { \alpha ^ { 2 } \left( \omega - \omega _ { i } \right) ^ { 4 } } } ,\tag{14}
$$

Comparing (14) with the corresponding equation for updating $\hat { u } _ { L }$ in VME reveals that the obtained Eq. (14) has an additional term in denominator related to the previously extracted modes which guarantees that the current obtained mode has no component at the center frequencies of previous modes. Similarly by minimizing (11) w.r.t. ωL, we will have:

$$
\omega _ { L } ^ { n + 1 } = \frac { \alpha \int _ { 0 } ^ { \infty } \omega \big | \hat { u } _ { L } ^ { n + 1 } ( \omega ) \big | ^ { 2 } d \omega - \frac { 1 } { \alpha ^ { 2 } } \int _ { 0 } ^ { \infty } \frac { 1 } { \left( \omega - \omega _ { L } ^ { n } \right) ^ { 5 } } \bigg | \ \hat { f } _ { u } ^ { n } ( \omega ) + \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } \left( \omega \right) \bigg | ^ { 2 } d \omega } { \alpha \int _ { 0 } ^ { \infty } \big | \hat { u } _ { L } ^ { n + 1 } ( \omega ) \big | ^ { 2 } d \omega } ,
$$

$$
\widehat { \lambda } ^ { n + 1 } = \widehat { \lambda } ^ { n } + \tau \left[ \widehat { f } ( \omega ) - \left( \widehat { u } _ { L } ^ { n + 1 } ( \omega ) + f _ { u } ^ { n + 1 } ( t ) + \sum _ { i = 1 } ^ { L - 1 } u _ { i } ^ { n + 1 } ( \omega ) \right) \right] ,\tag{19}
$$

where τ is the update parameter. Replacing $f _ { u } ( t )$ by (16) we will have:

and, by minimizing (11) w.r.t. $\hat { f } _ { u } ( \omega )$ . we will have:

(15)

$$
\widehat { f } _ { u } ^ { n + 1 } ( \omega ) = \frac { \displaystyle \alpha ^ { 2 } \big ( \omega - \omega _ { L } ^ { n + 1 } \big ) ^ { 4 } \bigg ( \widehat { f } ( \omega ) - \widehat { u } _ { L } ^ { n + 1 } ( \omega ) - \sum _ { i = 1 } ^ { L - 1 } \widehat { u } _ { i } ( \omega ) + \frac { \widehat { \lambda } ( \omega ) } { 2 } \bigg ) - \ \sum _ { i = 1 } ^ { L - 1 } \widehat { u } _ { i } ( \omega ) } { \displaystyle 1 + \alpha ^ { 2 } \big ( \omega - \omega _ { L } ^ { n + 1 } \big ) ^ { 4 } } .\tag{16}
$$

$$
\widehat { L } ^ { n + 1 } = \widehat { \lambda } ^ { n } + \tau \left[ \widehat { f } ( \omega ) - \left( \widehat { u } _ { l } ^ { n + 1 } ( \omega ) + \left[ \frac { \alpha ^ { 2 } \big ( \omega - \omega _ { l } ^ { n + 1 } \big ) ^ { 4 } \Big ( \widehat { f } ( \omega ) - \widehat { u } _ { l } ^ { n + 1 } ( \omega ) - \sum _ { i = 1 } ^ { L - 1 } \widehat { u } _ { i } ( \omega ) + \frac { \widehat { x } ( \omega ) } { 2 } \Big ) - \sum _ { i = 1 } ^ { L - 1 } \widehat { u } _ { i } ( \omega ) } { 1 + \alpha ^ { 2 } \big ( \omega - \omega _ { l } ^ { n + 1 } \big ) ^ { 4 } } \right] + \sum _ { i = 1 } ^ { L - 1 } u _ { i } ^ { n + 1 } ( \omega ) \right) \right] .\tag{20}
$$

Therefore, one iteration of SVMD is accomplished by Eqs. (17)– (20) sequentially. Accordingly, the complete algorithm for SVMD is summarized in Algorithm 1.

```latex
Algorithm 1 SVMD.
Input f(t)
set $\alpha , \epsilon _ { 1 } , \epsilon _ { 2 } ,$ and $\sigma ^ { 2 }$
Initialize, $L \gets 0$
repeat
$L \gets L + 1$
Initialize $\hat { u } _ { L } ^ { 1 } , \hat { \lambda } ^ { 1 } , ~ \omega _ { L } ^ { 1 } , n \gets 0$
repeat
n ← n + 1
1) Update uˆL for all $\omega \geq 0 \colon$
$\begin{array} { r } { \hat { u } _ { L } ^ { n + 1 } ( \omega ) = \frac { \hat { f } ( \omega ) + \alpha ^ { 2 } ( \omega - \omega _ { L } ^ { n } ) ^ { 4 } \hat { u } _ { L } ^ { n } ( \omega ) + \frac { \hat { \lambda } ( \omega ) } { 2 } } { [ 1 + \alpha ^ { 2 } ( \omega - \omega _ { L } ^ { n } ) ^ { 4 } ] [ 1 + 2 \alpha ( \omega - \omega _ { L } ^ { n } ) ^ { 2 } + \sum _ { i = 1 } ^ { L - 1 } \frac { 1 } { \alpha ^ { 2 } ( \omega - \omega _ { i } ) ^ { 4 } } ] } } \end{array}$ (21)
2) Update ωL :
$\begin{array} { r } { \omega _ { L } ^ { n + 1 } = \frac { \int _ { 0 } ^ { \infty } \omega \left| \hat { u } _ { L } ^ { n + 1 } ( \omega ) \right| ^ { 2 } d \omega } { \int _ { 0 } ^ { \infty } \left| \hat { u } _ { L } ^ { n + 1 } ( \omega ) \right| ^ { 2 } d \omega } } \end{array}$ (22)
3) Dual Ascent for all $\omega \geq 0 \colon$
λˆ n+1 $= \hat { \lambda } ^ { n } +$
$\tau [ \hat { f } ( \omega ) - ( \hat { u } _ { L } ^ { n + 1 } ( \omega ) + [ \frac { \alpha ^ { 2 } ( \omega - \omega _ { L } ^ { n + 1 } ) ^ { 4 } ( \hat { f } ( \omega ) - \hat { u } _ { L } ^ { n + 1 } ( \omega ) - \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } \ : ( \omega ) + \frac { \hat { x } ( \omega ) } { 2 } ) - \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } \ : ( \omega ) } { 1 + \alpha ^ { 2 } ( \omega - \omega _ { L } ^ { n + 1 } ) ^ { 4 } } ] + \sum _ { i = 1 } ^ { L - 1 } u _ { i } ^ { n + 1 } ( \omega ) ) ]$
(23)
until convergence: $\frac { \lVert \hat { u } _ { L } ^ { n + 1 } - \hat { u } _ { L } ^ { n } \rVert _ { 2 } ^ { 2 } } { \lVert \hat { u } _ { t } ^ { n } \rVert _ { 2 } ^ { 2 } } < \epsilon _ { 1 }$
until convergence: $\begin{array} { r } { \big | \sigma ^ { 2 } - \frac { \mathrm { i } } { T } \big | \big | ( f ( t ) - \sum \displaylimits _ { l = 1 : L } u _ { l } ( t ) ) \big | \big | _ { 2 } ^ { 2 } \big | / \sigma ^ { 2 } \ < \epsilon _ { 2 } , } \end{array}$
```

It can be seen that SVMD solves K optimization problems, each looking for one mode, whereas VMD solves one optimization problem looking for K modes. In other words, the optimization problem in SVMD can approximately be considered as K one-dimensional optimization problems at each frequency whereas VMD is one Kdimensional optimization problem. This results in a much more iterations till convergence for VMD in comparison with SVMD, as our simulations verify this.

The SVMD algorithm represented here (Algorithm 1) assumes that a prior approximate value of the power of the additive white noise $( \sigma ^ { 2 } )$ is known and the algorithm terminates its search for new modes until the power of un-processed part of the signal is almost equal to the power of noise. However, if the number of modes is known, the end criterion of the algorithm can simply be set to extract all modes. Also, the sensitivity of the power of the last mode with respect to the center frequency can be considered as another criterion to cease the algorithm.

## 2.2. SVMD with varying α

One of the most important parameters of SVMD is α. High value of α may result in some wrong modes which are actually noise or may cause convergence problems. On the other hand, small value of α may lead to the mode mixing problems, in which two or more modes are considered as one. Therefore, proper determination of α is a compromising task. It is also true for VMD. In this section, we introduce a simple heuristic method to change α in each iteration to avoid the problems related to low or high value of α. In this method, the algorithm starts with a very low α and as the iteration proceeds, α exponentially grows up not exceeding a maximum allowable value determined by the user. By this variations of $\alpha ,$ the algorithm tends to find the strongest mode among the modes not found before. Consequently, the modes are extracted almost in order of their powers. Our investigations have shown that this modification on SVMD diminishes the problems encountered by both high and low values of α mentioned before, whereas it obviates the need for proper setting of α. This SVMD with varying α is described in Algorithm 2.

Algorithm 2 SVMD with varying α.   
Input f(t)   
set $\alpha _ { m i n } , \alpha _ { m a x } , \epsilon _ { 1 } , \epsilon _ { 2 } ,$ and $\sigma ^ { 2 }$   
Initialize $L \gets 0$   
repeat   
$L \gets L + 1$   
set $\hat { u } _ { L } ^ { 1 } , \hat { \lambda } ^ { 1 } , ~ n \gets 0 , ~ m \gets 0$   
$\alpha _ { 1 }  \alpha _ { m i n }$   
$\omega _ { L } ^ { 1 } $ Initialize to 0 or by a random value between 0 and π   
repeat   
$m \gets m + 1$   
repeat   
$n \gets n + 1$   
1) Update $\hat { u } _ { L }$ for all ω ≥ 0:   
$\begin{array} { r } { \hat { u } _ { L } ^ { n + 1 } ( \omega ) = \frac { \hat { f } ( \omega ) + \alpha _ { m } ^ { 2 } ( \omega - \omega _ { L } ^ { n } ) ^ { 4 } \hat { u } _ { L } ^ { n } ( \omega ) + \frac { \hat { \lambda } ( \omega ) } { 2 } } { [ 1 + \alpha _ { m } ^ { 2 } ( \omega - \omega _ { L } ^ { n } ) ^ { 4 } ] [ 1 + 2 \alpha _ { m } ( \omega - \omega _ { L } ^ { n } ) ^ { 2 } + \sum _ { i = 1 } ^ { L - 1 } \frac { 1 } { \alpha _ { m } ^ { 2 } ( \omega - \omega _ { i } ) ^ { 4 } } ] } } \end{array}$ (24) (24   
2) Update ωL :   
$\begin{array} { r } { \omega _ { L } ^ { n + 1 } = \frac { \int _ { 0 } ^ { \infty } \omega \left| \hat { u } _ { L } ^ { n + 1 } ( \omega ) \right| ^ { 2 } d \omega } { \int _ { 0 } ^ { \infty } \left| \hat { u } _ { L } ^ { n + 1 } ( \omega ) \right| ^ { 2 } d \omega } } \end{array}$ (25)   
3) Dual Ascent for all ω ≥ 0:   
$\begin{array} { r } { \hat { \lambda } ^ { n + 1 } = \hat { \lambda } ^ { n } + \tau [ \hat { f } ( \omega ) - ( \hat { u } _ { L } ^ { n + 1 } ( \omega ) + \big [ \frac { \alpha _ { m } ^ { 2 } ( \omega - \omega _ { L } ^ { n + 1 } ) ^ { 4 } ( \hat { f } ( \omega ) - \hat { u } _ { L } ^ { n + 1 } ( \omega ) - \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } ( \omega ) + \frac { \hat { x } ( \omega ) } { 2 } ) - \sum _ { i = 1 } ^ { L - 1 } \hat { u } _ { i } ( \omega ) } { 1 + \alpha _ { m } ^ { 2 } ( \omega - \omega _ { L } ^ { n + 1 } ) ^ { 4 } } \big ] + } \end{array}$   
$\begin{array} { r } { \sum _ { i = 1 } ^ { L - 1 } u _ { i } ^ { n + 1 } ( \omega ) ) ] } \end{array}$   
(26)   
until convergence: $\frac { \lVert \hat { u } _ { L } ^ { n + 1 } - \hat { u } _ { L } ^ { n } \rVert _ { 2 } ^ { 2 } } { \lVert \hat { u } _ { L } ^ { n } \rVert _ { 2 } ^ { 2 } } < \epsilon _ { 1 } .$   
set uˆ1L $, \hat { \lambda } ^ { 1 } , n \gets$ 0 and $\omega _ { L } ^ { 1 }  \omega _ { L } ^ { n + 1 }$   
$\alpha _ { m + 1 }  2 \alpha _ { m }$   
until $\alpha _ { m + 1 } \leq \alpha _ { m a x }$   
until convergence: $\begin{array} { r } { \overrightarrow { \vert \sigma ^ { 2 } } - \frac { 1 } { T } \Vert ( f ( t ) - \sum _ { l = 1 : L } u _ { l } ( t ) ) \Vert _ { 2 } ^ { 2 } \vert / \sigma ^ { 2 } < \epsilon _ { 2 } , } \end{array}$

## 3. Simulation

In this section, we evaluate the proposed SVMD algorithm through applying it to a series of test signals and compare its performance against VMD. These signals were almost similar to those presented in [15] and [29], containing both simple and relatively complex structures. They contain fixed-frequency components, the components with rapid instantaneous change from low frequency to the high frequency variations, intrawave frequency modulation and chirp signal. All algorithms are implemented in MATLAB R2012a.

## 3.1. Setting the Parameters in VMD

As completely described in [15] and [29], in VMD, the value of the total number of modes K, and the weighting factor α, should properly be set. With assuming P as the true number of modes of the signal, optimum results will be obtained when K=P. Therefore, in all simulations of VMD, we assumed that the value of P is known and K=P and also, α was chosen such that the best performance of VMD is obtained.

<!-- image-->  
(a)

<!-- image-->

<!-- image-->

<!-- image-->

(d）  
<!-- image-->  
（e）  
Fig. 1. a) Artificial signal $f _ { s i g 1 } ( t ) \mathrm { ~ b ) }$ Its temporal constituent components c) Its spectrum of decomposed modes of signal, d) Obtained modes using SVMD, and e) Decomposed modes using VMD.

As in [15], random initialization for the center frequencies of modes $( \omega _ { k } { ' s } )$ and zero initialization for modes $u _ { k }$ is done. Also, the Lagrangian multipliers can result in the complete reconstruction of the input signal at low noise levels, but with more important noise they become an obstacle for convergence [15]. Thus, the update parameter of λ namely τ was zero, as we assume that the signal is affected by noise.

## 3.2. Setting the parameters in SVMD

In all simulations, we have implemented the proposed method through Algorithm 2, SVMD with varying α. Therefore, it only needs to set a starting value $( \alpha _ { \mathrm { m i n } } )$ and a maximum allowable value $\left( \alpha _ { \mathrm { m a x } } \right)$ for α, which we have set them to 10 and 20000, respectively.

## 3.3. Artificial Signals and Simulation Results

In this subsection, we compare the results of VMD and SVMD applied on four artificial signals. We applied the decomposition methods after contaminating the signals by additive white Gaussian noise with σ =0.2. The first signal has intrawave frequency modulation as follows:

$$
\begin{array} { l } { { f _ { S i g 1 } ( t ) = \displaystyle \frac { 1 } { 1 . 2 + C o s ( 2 \pi t ) } } } \\ { { \displaystyle \qquad + \Biggl [ \frac { 1 } { 1 . 5 + S i n ( 2 \pi t ) } \times C o s ( 4 8 \pi t + 0 . 2 C o s ( 9 6 \pi t ) ) \Biggr ] . } } \end{array}\tag{28}
$$

The signal and its modes are depicted in Fig. 1. The first mode is a bell-shaped component and has basically low-pass nature, whereas the second mode has clearly intrawave frequency modulation with a main peak at 48π . This phenomena makes an impressive amount of higher-order harmonics on the spectrum. As it is clear, this nature of second component is clearly in contradiction to the narrow-band assumption in basis of VMD and SVMD, and as it is expected, these methods are in trouble with extraction of the second mode. The results of VMD and SVMD are also shown in Fig. 1 and it can be seen that the higher order harmonics are not completely ascribed to the second mode. In fact, they are shared between both modes and creating some small ripples in the first mode. It is noteworthy that the two methods have obtained the same results.

The second artificial signal is composed of three frequency components with one constant frequency component and two AMmodulated components:

$$
\begin{array} { l } { { f _ { S i g 2 } ( t ) = 2 C o s ( 4 \pi t ) + \displaystyle \left( \frac { 1 + C o s ( 2 \pi t ) } { 2 } \times C o s ( 4 8 \pi t ) \right) } } \\ { { + \displaystyle \Biggl [ \frac { 1 + s i n ( 2 \pi t ) } { 2 } \times C o s ( 9 6 \pi t ) \Biggr ] . } } \end{array}\tag{29}
$$

which is shown in Fig. 2 with its spectrum and modes. The result obtained by SVMD and VMD methods can also be seen in Fig. 2, which shows that each mode is clearly located around its related center frequency. The results indicate nice separation between three constituent components of signal. Also, it confirms that VMD and SVMD obtain the same modes.

And finally the third artificial signal is sum of a chirp signal and a cosine waveform with rapid transition between two constant frequencies:

$$
f _ { S i g 3 } ( t ) = 2 C o s \bigl ( 1 0 \pi t + 1 0 \pi t ^ { 2 } \bigr ) + \left\{ \begin{array} { l l } { C o s ( 6 0 \pi t ) } & { t \leq 0 . 5 } \\ { C o s ( 1 0 0 \pi t - 1 0 \pi ) } & { t > 0 . 5 } \end{array} \right.\tag{30}
$$

<!-- image-->

(a)  
<!-- image-->  
（c）

<!-- image-->  
(b)

<!-- image-->  
(d）

<!-- image-->  
(e）  
Fig. 2. a) Artificial signal $f _ { s i g 2 } ( t )$ b) Its temporal constituent components c) Its spectrum of decomposed modes of signal, d) Obtained modes using SVMD, and e) Decomposed modes using VMD.

In this example, the main challenge is the wide spectrum of the chirp signal which is in conflict with narrow-band assumption. As we let t ∈ [0, 1], the chirp’s instantaneous frequency spreads on the spectrum between 10π and 30π and thus, its theoretical center frequency is 20π. The modes obtained by VMD and SVMD are shown in Fig. 3, which emphasizes both methods result in the same modes. Since the second component of signal is bi-harmonic, both algorithms assign each half of the piecewise-constant frequency signal to a separate mode.

Finally, we have compared the sensitivity of VMD and SVMD to the various methods for initializing the center frequencies. In this regard, we examined both the methods through the three different ways: uniform initialization, zero initialization and the random initialization [15]. To this end, we considered the following tri-harmonic signal:

$$
f _ { s i g 4 } ( t ) = C o s ( 4 \pi t ) + \frac { 1 } { 4 } C o s ( 4 8 \pi t ) + \frac { 1 } { 1 6 } C o s ( 5 7 6 \pi t ) ,\tag{31}
$$

and then ran both the SVMD and VMD methods 100 times with different methods of center frequency initialization. The trend of converged center frequencies have been presented in Fig. 4, which emphasizes that the SVMD (Fig. 4(b)) has completely converged to the true modes; whereas VMD (Fig. 4(a)) performs some failures. This investigation indicates that SVMD has more robustness to the initialization of the center frequencies than VMD.

## 3.4. Real World Signals and Simulation Results

In this section, some examinations were performed on a real world electrocardiogram (ECG) signal. ECG signal may be affected by some noise sources [31] and mode decomposition techniques can be used for noise suppression. The ECG signals prepared from MIMIC archive bank of Physionet databases, including biological signals recorded from patients of CCU and ICU departments in one of the hospitals in Boston of America1. Here, we added a white Gaussian noise with 10dB SNR to the input signals and then performed decomposition process using SVMD and also VMD on 10 ECG signals randomly selected from dataset. The modes obtained by applying SVMD on data 039m is represented in Fig. 5 as an example. As expected, SVMD was capable of recovering all the determinant modes in the ECG signal. To quantitatively compare the denoising capability of SVMD and VMD, the correlation coefficient (CC) between the reconstructed signal (sum of the extracted modes) and the original noise free ECG signal were calculated for two methods and the results are summarized in Table 1. The value of CC close to 1 indicates good denoising capability. However, because of the presence of noise, CC is always smaller than 1. According to the table, VMD and SVMD have almost the same value of CC with a bit superiority of SVMD.

One of the main advantages of SVMD over VMD is that it has no need to know the number of modes. As explained before, if the number of modes K is not properly set for VMD, some problems in the mode decomposition would be raised. The ECG signals selected for our evaluations contain a variety of modes, from 8 to 37, and the results mentioned in Table 1 were obtained by running VMD with the best choice of K for each signal. In fact, for each signal, we applied VMD several times with different values of K and the result obtaining the largest value of CC was considered as the best K. As the table shows, SVMD as a method with no need to know the number of modes, yields CC values comparable to the highest ones obtainable by VMD.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

Fig. 3. a) Artificial signal $f _ { s i g 3 } ( t )$ b) Its temporal constituent components c) Its spectrum of decomposed modes of signal, d) Obtained modes using SVMD, and e) Decomposed modes using VMD.  
<!-- image-->  
(a)

<!-- image-->  
Fig. 4. Sensitivity of VMD (a) and SVMD (b) to the initialization of center frequencies. For each decomposition method, the noisy signal $f _ { s i g 4 } ( t )$ with three modes was decomposed 100 times with different initial center frequencies. The dashed vertical lines indicate the position of true center frequencies and the color plots show the variations of center frequencies from initializations to the final converged values. The black circles in (a) demonstrate that the VMD sometimes does not converge to the true center frequencies.

Another advantage of SVMD over VMD is its lower computational complexity, especially when K is high. As explained before,

SVMD converges to the modes through lower iterations than VMD, as it was seen in Fig. 4. To more precisely compare the SVMD and VMD methods in computational complexity aspect, the number of multiplications done in decomposing an 8-s segment of each of the selected ECG signals were counted and their values are given in Table 1. These results are also summarized in Fig. 6, which shows the number of multiplications versus the number of modes available in signals. All the results emphasize that the computational complexity of SVMD is much (approximately, 10 times in average) lower than that of VMD.

<!-- image-->

<!-- image-->

<!-- image-->  
（c）

<!-- image-->  
(d）

<!-- image-->  
(e)

<!-- image-->

<!-- image-->  
(f)

(g）  
<!-- image-->  
(i  
Fig. 5. a) Data number 039m (clean signal), b) Noisy signal, c) Sum of 17 recovered modes, d-i) The five first modes obtained by SVMD.

Table 1  
Computational burden and accuracy of reconstructed signals:comparison of VMD and SVMD methods in the number of multiplications and correlation coefficients for 10 ECG signals.
<table><tr><td rowspan="2">Data SVMD number</td><td colspan="3"></td><td colspan="3">VMD (with choosing best K)</td></tr><tr><td>Number of obtained modes (L)</td><td>CC</td><td>Computation burden (Number of multiplications)</td><td>K</td><td>Cc</td><td>Computation burden (Number of multiplications)</td></tr><tr><td>452m</td><td>8</td><td>0.9989</td><td>35,100</td><td>8</td><td>0.9925</td><td>20,160</td></tr><tr><td>213m</td><td>12</td><td>0.9992</td><td>83.472</td><td>12</td><td>0.9989</td><td>305,160</td></tr><tr><td>222m</td><td>15</td><td>0.9935</td><td>128,320</td><td>15</td><td>0.9908</td><td>130,950</td></tr><tr><td>039m</td><td>17</td><td>0.9919</td><td>141,092</td><td>18</td><td>0.9903</td><td>430,020</td></tr><tr><td>438m</td><td>20</td><td>0.9867</td><td>190,120</td><td>21</td><td>0.9871</td><td>820,640</td></tr><tr><td>226m</td><td>23</td><td>0.9895</td><td>273,171</td><td>23</td><td>0.9867</td><td>3,478.060</td></tr><tr><td>055m</td><td>25</td><td>0.9995</td><td>228,825</td><td>26</td><td>0.9993</td><td>2,465,320</td></tr><tr><td>476m</td><td>25</td><td>0.9889</td><td>307,925</td><td>26</td><td>0.9891</td><td>3,598,140</td></tr><tr><td>225m 444m</td><td>27 36</td><td>0.9878</td><td>295,596</td><td>28</td><td>0.9880</td><td>4,289,600</td></tr><tr><td></td><td></td><td>0.9990</td><td>436,248</td><td>37</td><td>0.9995</td><td>5,832,000</td></tr><tr><td>Mean (and Standard deviation)</td><td></td><td>0.9935 (0.0052)</td><td>211,990 (120.570)</td><td>1</td><td>0.9922 (0051)</td><td>2,137,005 (2,077,300)</td></tr></table>

<!-- image-->  
Fig. 6. Computational complexity versus number of modes derived from Table 1.

## 4. Conclusion

In this paper, we have considered the signal decomposition issue and presented an iterative method for decomposing a signal into constituent components. The method named as SVMD, and can be viewed as a successive implementation of VMD or as an extension of VME approach. The method was established by adding some criteria to the VMD algorithm which guarantee the newest mode is different from the modes previously found. We have shown through some simulations that the method almost converges to the same modes as VMD does. The most advantage of SVMD over VMD is that it does not need to know the number of modes available in the signal, which is a crucial parameter for VMD. In other words, SVMD extracts modes one after the other until the reconstruction error reaches to a threshold determined by the user. Moreover, SVMD has lower computational complexity compared with VMD. Besides, we have shown that SVMD is less sensitive to the initial values of the center frequencies of the modes, in comparison to VMD. Our examinations have shown that by randomly initializing or even initializing each center frequency to zero, SVMD successfully converges to the true modes. On the other hand, some initializations make VMD converge to the wrong or double modes, as it was also mentioned in [15].

Finally, it seems that the idea introduced here to convert simultaneous extraction into the successive extraction can be used for 2-D VMD, CVMD, and VNCMD algorithms and consequently lower their computational complexities, obviate the need for prior knowledge about K, and reduce the sensitivity to the center frequency initialization.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Appendix A

In this appendix, we show that the modes obtained by SVMD are almost the same as VMD. Assume that the algorithm of SVMD is converged to the true modes and also, the modes are spectrally distinct. Thus $\hat { u } _ { L } ( \omega ) = \hat { u } _ { \scriptscriptstyle L } ^ { n + 1 } ( \omega ) = \hat { u } _ { \scriptscriptstyle L } ^ { n } ( \omega )$ and by assuming ωL is the true center frequency of $\hat { u } _ { L } ( \omega )$ , Eq. (21) results in:

uˆL (ω)

$$
= \frac { \hat { f } ( \omega ) } { 1 + 2 \alpha \left( \omega - \omega _ { L } \right) ^ { 2 } + 2 \alpha ^ { 3 } \left( \omega - \omega _ { L } \right) ^ { 6 } + \left( \frac { 1 } { \alpha ^ { 2 } } + \left( \omega - \omega _ { L } \right) ^ { 4 } \right) \left( \sum _ { i = 1 } ^ { L - 1 } \frac { 1 } { \left( \omega - \omega _ { i } \right) ^ { 4 } } \right) }\tag{A.1}
$$

where for simplicity we assumed $\widehat { \lambda } ( \omega ) = 0 .$ This equation shows that the mode is compact around $\omega _ { L }$ and its value rapidly tends to zero by distancing from $\omega _ { L }$ . For ω near $\omega _ { L } ,$ , the values of $\{ \frac { 1 } { \alpha ^ { 2 } { \bf \Gamma } ( \omega - \omega _ { i } ) ^ { 4 } } \} _ { i = 1 : L - 1 }$ are very small by assuming distinct center frequencies and large value of $\alpha .$ Therefore, $\hat { u } _ { L } ( \omega )$ can well be approximated as:

$$
\hat { u } _ { L } ( \omega ) \cong \frac { \hat { f } ( \omega ) } { 1 + 2 \alpha \left( \omega - \omega _ { L } \right) ^ { 2 } }\tag{A.2}
$$

By the same assumptions, it can be seen that VMD is also converged to the mode described by the above equation. To that end, by assuming the convergence is occurred, Eq. (27) of [15] can be represented as:

$$
\hat { u } _ { L } ( \omega ) = \frac { \hat { f } ( \omega ) - \sum _ { 0 \le \overset { i \le ~ } { i \neq } } \kappa \hat { u } _ { i } ( \omega ) } { 1 + 2 \alpha { ( \omega - \omega _ { L } ) } ^ { 2 } }\tag{A.3}
$$

which is, by the assumption of distinct modes, approximated as the same mode obtained by A.2.

## CRediT authorship contribution statement

Mojtaba Nazari: Conceptualization, Software, Validation, Investigation, Writing - original draft, Methodology. Sayed Mahmoud Sakhaei: Conceptualization, Methodology, Formal analysis, Writing - review & editing, Supervision, Project administration.

## References

[1] N.E. Huang, Z. Shen, S.R. Long, M.C. Wu, H.H. Shih, Q. Zheng, N.-C. Yen, C.C. Tung, H.H. Liu, The empirical mode decomposition and the Hilbert spectrum for non-linear and non-stationary time series analysis, Proc. R. Soc. 454 (1971) (Mar. 1998) 903–995.

[2] M. Blanco-Velasco, B. Weng, K.E. Barner, ECG signal denoising and baseline wander correction based on the empirical mode decomposition, Comput. Biol. Med. 38 (1) (2008) 1–13.

[3] D. Labate, F. La Foresta, G. Occhiuto, F. Carlo Morabito, A. Lay-Ekuakille, P. Vergallo, Empirical mode decomposition vs. wavelet decomposition for the extraction of respiratory signal from single-channel ECG: a comparison, IEEE Sens. J. 13 (7) (July 2013).

[4] A.O. Andrade, S. Nasuto, P. Kyberd, C.M. Sweeney-Reed, F. Van Kanijn, EMG signal filtering based on empirical mode decomposition, Biomed. Signal Process. Control 1 (1) (Jan. 2006) 44–55.

[5] Z.H. Slimane, A. Naït-Ali, QRS complex detection using empirical mode decomposition, Digital Signal Process. 20 (4) (2010) 1221–1228.

[6] S. Mishra, D. Das, R. Kumar, P. Sumathi, A power-line interference canceler based on sliding DFT phase locking scheme for ECG signals, IEEE Trans. Instrum. Meas. 64 (1) (Jan. 2015) 132–142.

[7] I. Mostafanezhad, O. Boric-Lubecke, V. Lubecke, D.P. Mandic, Application of empirical mode decomposition in removing fidgeting interference in Doppler radar life signs monitoring devices, in: Proc. IEEE Eng. Med. Biol. Conf. (EMBC), Jan. 2009, pp. 340–343.

[8] B. Barnhart, W. Eichinger, Empirical mode decomposition applied to solar irradiance, global temperature, sunspot number, and CO2 con-centration data, J. Atmospher. Solar-Terrestrial Phys. 73 (13) (Aug. 2011) 1771–1779.

[9] T. Lee, T.B.M.J. Ouarda, Prediction of climate nonstationary oscillation processes with empirical mode decomposition, J. Geophys. Res. 116 (D6) (2011).

[10] X. Hu, S. Peng, W.L. Hwang, EMD Revisited: a new understanding of the envelope and resolving the mode mixing problem in AM-FM signals, IEEE Trans. Signal Process. 60 (3) (Mar. 2012) 1075–1086.

[11] S. Meignen, V. Perrier, A new formulation for empirical mode decomposition based on constrained optimization, IEEE Signal Process. Lett. 14 (12) (Dec. 2007) 932–935.

[12] T. Oberlin, S. Meignen, V. Perrier, An alternative formulation for the empirical mode decomposition, IEEE Trans. Signal Process. 60 (5) (May. 2012) 2236–2246.

[13] G. Rilling, P. Flandrin, One or two frequencies? The empirical mode decomposition answers, IEEE Trans. Signal Process. 56 (1) (Jan. 2008) 85–95.

[14] Z. Wu, N.E. Huang, Ensemble empirical mode decomposition: a noise-assisted data analysis method, Adv. Adapt. Data Anal. 01 (01) (Jan. 2009) 1–41.

[15] K. Dragomiretskiy, D. Zosso, Variational mode decomposition, IEEE Trans. Signal Process. 62 (3) (Feb. 2014) 531–544.

[16] Y. Wang, R. Markert, J. Xiang, W. Zheng, Research on variational mode decomposition and its application in detecting rub-impact fault of the rotor system, Mech. Syst. Signal Process. 60–61 (Aug. 2015) 243–251.

[17] M. Zhang, Z. Jiang, K. Feng, Research on variational mode decomposition in rolling bearings fault diagnosis of the multistage centrifugal pump, Mech. Syst. Signal Process. 93 (Sep. 2017) 460–493.

[18] S. Lahmiri, Comparative study of ECG signal denoising by wavelet thresholding in empirical and variational mode decomposition domains, IEEE Healthcare Technol. Lett. 1 (Issue 3) (Sep. 2014) 104–109.

[19] F. Li, B. Zhang, S. Verma, K.J. Marfurt, Seismic signal denoising using thresholded variational mode decomposition, Explor. Geophy. 49 (4) (2017) 450–461.

[20] C. Dora, P.K. Biswal, An improved algorithm for efficient ocular artifact suppression from frontal EEG electrodes using VMD, Biocybernet. Biomed. Eng. (2019).

[21] A. Upadhyay, R.B. Pachori, Instantaneous voiced/non-voiced detection in speech signals based on variational mode decomposition, J. Franklin Inst. 352 (Issue 7) (July 2015) 2679–2707.

[22] W. Liu, S. Cao, Y. Chen, Applications of variational mode decomposition in seismic time-frequency analysis, Geophysics 81 (5) (2016) V365–V378.

[23] F. Li, B. Zhang, R. Zhai, H. Zhou, K.J. Marfurt, Depositional sequence characterization based on seismic variational mode decomposition, Interpretation 5 (2) (2017) SE97–SE106.

[24] Y. Liu, G. Yanga, M. Li, H. Yin, Variational mode decomposition denoising combined the detrended fluctuation analysis, Signal Process. 125 (Aug. 2016) 349–364.

[25] K. Dragomiretskiy, D. Zosso, Two-dimensional variational mode decomposition, in: Energy Minimization Methods in Computer Vision and Pattern Recognition, Springer, Cham, Switzerland, 2015, pp. 197–208.

[26] Y. Wang, F. Liu, Z. Jiang, S. He, Q. Mo, Complex variational mode decomposition for signal processing applications, Mech. Syst. Signal Process. 86 (Part A) (Mar. 2017) 75–85.

[27] S. Yu, J. Ma, Complex variational mode decomposition for slop-preserving denoising, IEEE Trans. Geosci. Remote Sens. 56 (issue 1) (Jan. 2018) 586–597.

[28] S. Chen, X. Dong, Z. Peng, W. Zhang, G. Meng, Nonlinear chirp mode decomposition: a variational method, IEEE Trans. Signal Process. 65 (22) (Nov. 2017) 6024–6037.

[29] M. Nazari, S.M. Sakhaei, Variational mode extraction: an efficient method for single-lead ECG-derived respiration, IEEE J. Biomed. Health Inf. 22 (4) (July 2018) 1059–1067.

[30] D.P. Bertsekas, Constrained optimization and lagrange multiplier methods, Computer Science and Applied Mathematics, 1, Academic, Boston, MA, USA, 1982.

[31] G.M. Friesen, T.C. Jannett, M.A. Jadallah, S.L. Yates, S.R. Quint, H.T. Nagle, A comparison of the noise sensitivity of nine qrs detection algorithms, IEEE Trans. Biomed. Eng. 37 (1) (Jan. 1990) 85–98.