# Path: app/furuta_pendulum/FURUTA_PENDULUM/ode.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .base import FURUTA_PENDULUM


def ode(self: "FURUTA_PENDULUM", x: np.ndarray, input: float, param: np.ndarray) -> np.ndarray:
    """
    MATLAB の private メソッド ode(obj,in1,input,in3) に相当。

    ここに、Symbolic Math Toolbox が生成した t2=np.cos(p) ... の超長い式を
    Python に書き換えて貼り付けることができます。

    現在は接続確認用にかなり単純化したモデルを使っています。
    """
    p, th, dp, dth = x
    (
        m1,
        m2,
        J,
        jx,
        jy,
        jz,
        L,
        lg,
        Dp,
        Dth,
        gravity,
        a,
    ) = param
    
    t2 = np.cos(p)
    t3 = np.cos(th)
    t4 = np.sin(p)
    t5 = np.sin(th)
    t6 = Dp*dp
    t7 = Dth*dth
    t8 = L**2
    t9 = dp**2
    t10 = dth**2
    t11 = lg**2
    t13 = m2**2
    t14 = p*2.0
    t15 = th*2.0
    t16 = J*jz*4.0
    t20 = jx*jz*4.0
    t12 = t11**2
    t17 = t2**2
    t18 = t2**3
    t21 = t3**2
    t22 = t3**3
    t24 = np.sin(t14)
    t25 = t4**2
    t26 = t4**3
    t27 = np.sin(t15)
    t28 = t5**2
    t29 = -t7
    t30 = L*lg*m2*t3
    t31 = gravity*lg*m2*t5
    t32 = jz*m1*t8
    t33 = jx*m2*t11*4.0
    t34 = jz*m2*t11*4.0
    t35 = jy*t4*t10
    t36 = jz*t4*t10
    t37 = J*m2*t11*4.0
    t38 = jz*m2*t8*4.0
    t49 = jx*t3*t4*t5
    t50 = jz*t3*t4*t5
    t56 = L*lg*m2*t5*t10
    t63 = m1*m2*t8*t11
    t72 = jx*t2*t5*t9*2.0
    t73 = jy*t2*t5*t9*2.0
    t90 = jz*t2*t3*t5*t9
    t91 = dp*dth*jx*t2*t3*t4*4.0
    t92 = dp*dth*jy*t2*t3*t4*4.0
    t105 = t8*t11*t13*4.0
    t120 = jx*t2*t3*t4*t10*2.0
    t124 = jx*t2*t3*t5*t9*3.0
    t163 = jy*t2*t3*t5*t9*-2.0
    t19 = t17**2
    t23 = t21**2
    t40 = J*jy*t17*4.0
    t42 = t16*t17
    t43 = J*jx*t21*4.0
    t44 = t16*t21
    t45 = jx*jy*t17*4.0
    t47 = jy*jz*t17*4.0
    t51 = dp*dth*jy*t24
    t52 = dp*dth*jz*t24
    t53 = dp*dth*jx*t27
    t54 = dp*dth*jz*t27
    t55 = t12*t13*4.0
    t58 = J*jz*t17*-4.0
    t59 = J*jz*t21*-4.0
    t60 = jx*jz*t17*8.0
    t62 = -t36
    t64 = jx*t10*t26
    t65 = jy*t10*t26
    t67 = jy*m1*t8*t17
    t69 = t17*t32
    t70 = jx*m1*t8*t21
    t71 = t21*t32
    t75 = -t50
    t76 = jx*t4*t5*t17
    t77 = jy*t4*t5*t17
    t81 = jy*m2*t8*t17*4.0
    t83 = t17*t38
    t84 = jx*m2*t8*t21*4.0
    t85 = t21*t38
    t86 = J*jx*t3*t17*8.0
    t88 = J*jy*t3*t17*8.0
    t93 = t17*t33
    t96 = t17*t34
    t98 = t21*t34
    t102 = jy*jz*t3*t17*8.0
    t104 = -t73
    t106 = dp*dth*m2*t11*t27
    t107 = jx*t5*t10*t17
    t109 = jy*t5*t10*t17
    t111 = dp*dth*jx*t4*t18*4.0
    t112 = dp*dth*jy*t4*t18*4.0
    t117 = jz*m2*t8*t17*-4.0
    t119 = jz*m2*t8*t21*-4.0
    t121 = t2*t3*t35*2.0
    t125 = t3*t73
    t126 = -t91
    t128 = jx*m2*t11*t17*-4.0
    t130 = jy*m2*t11*t17*8.0
    t131 = jz*m2*t11*t17*-4.0
    t133 = jz*m2*t11*t21*-4.0
    t134 = jx*jz*t3*t17*-8.0
    t144 = t17*t49
    t146 = jx*t4*t10*t18*2.0
    t147 = t18*t35*2.0
    t148 = jx*t5*t9*t18*3.0
    t149 = jy*t5*t9*t18*3.0
    t150 = jx*t4*t10*t21*2.0
    t151 = t21*t36*2.0
    t154 = (jx*t9*t24)/2.0
    t155 = (jy*t9*t24)/2.0
    t156 = (jy*t10*t24)/2.0
    t157 = (jz*t10*t24)/2.0
    t158 = (jx*t9*t27)/2.0
    t159 = (jz*t9*t27)/2.0
    t160 = (jx*t10*t27)/2.0
    t161 = (jz*t10*t27)/2.0
    t162 = -t120
    t164 = -t90
    t165 = t12*t13*t21*-4.0
    t166 = jx*t2*t4*t9*t21
    t167 = jy*t2*t4*t9*t21
    t168 = jx*t3*t4*t10*t17
    t169 = t2*t21*t35
    t170 = t3*t17*t35
    t171 = t2*t21*t36
    t172 = J*jx*t17*t21*8.0
    t174 = jx*t3*t5*t9*t17
    t175 = jy*t3*t5*t9*t17
    t179 = jz*t3*t5*t10*t17
    t180 = dp*dth*jx*t2*t4*t21*4.0
    t181 = dp*dth*jy*t2*t4*t21*2.0
    t182 = dp*dth*jz*t2*t4*t21*2.0
    t183 = dp*dth*jx*t3*t5*t17*2.0
    t184 = dp*dth*jy*t3*t5*t17*2.0
    t185 = jx*jy*t17*t21*-4.0
    t196 = jx*m1*t3*t8*t17*2.0
    t200 = jx*m2*t3*t11*t17*8.0
    t209 = jx*t3*t4*t10*t18*4.0
    t210 = t3*t18*t35*4.0
    t216 = dp*dth*jx*t3*t4*t18*8.0
    t217 = dp*dth*jy*t3*t4*t18*8.0
    t221 = jx*m2*t3*t8*t17*8.0
    t224 = jy*m2*t3*t8*t17*8.0
    t235 = jy*m2*t11*t17*t21*4.0
    t240 = jx*t2*t4*t10*t21*-2.0
    t244 = jx*t3*t5*t9*t18*-3.0
    t248 = (m2*t9*t11*t27)/2.0
    t249 = t21*t105
    t254 = L*jx*lg*m2*t4*t5*t21*8.0
    t255 = L*jz*lg*m2*t4*t5*t21*8.0
    t257 = jx*m2*t8*t17*t21*8.0
    t262 = jx*m2*t11*t17*t22*8.0
    t270 = jz*m2*t11*t17*t21*8.0
    t271 = t8*t11*t13*t21*-4.0
    t274 = t17*t21*t35*2.0
    t276 = dp*dth*jy*t4*t18*t21*-4.0
    t280 = jy*m2*t11*t17*t22*-8.0
    t284 = t18*t21*t35*-2.0
    t39 = J*jx*t19*4.0
    t41 = J*jy*t19*4.0
    t46 = t19*t20
    t48 = jy*jz*t19*4.0
    t66 = jx*m1*t8*t19
    t68 = jy*m1*t8*t19
    t74 = -t60
    t78 = -t52
    t79 = -t54
    t80 = jx*m2*t8*t19*4.0
    t82 = jy*m2*t8*t19*4.0
    t87 = J*jx*t3*t19*8.0
    t89 = J*jy*t3*t19*8.0
    t94 = t19*t33
    t95 = jy*m2*t11*t19*4.0
    t97 = t23*t33
    t99 = t23*t34
    t100 = t3*t60
    t101 = jx*jz*t3*t19*8.0
    t103 = jy*jz*t3*t19*8.0
    t108 = jx*t5*t10*t19
    t110 = jy*t5*t10*t19
    t113 = -t65
    t116 = -t69
    t118 = -t71
    t122 = -t86
    t127 = t21*t55
    t132 = jx*m2*t11*t23*-4.0
    t137 = t21*t40
    t139 = t21*t42
    t140 = t21*t45
    t142 = t21*t47
    t145 = t3*t77
    t152 = -t112
    t153 = -t76
    t173 = J*jy*t19*t21*-4.0
    t177 = t3*t109
    t186 = jy*jz*t19*t21*-4.0
    t187 = -t147
    t188 = -t148
    t189 = -t107
    t190 = -t150
    t192 = -t155
    t193 = -t157
    t194 = -t159
    t195 = -t161
    t198 = t3*t67*2.0
    t201 = jx*m2*t3*t11*t19*8.0
    t202 = t3*t130
    t203 = jy*m2*t3*t11*t19*8.0
    t205 = t21*t67
    t207 = t21*t69
    t208 = t2*t150
    t211 = -t172
    t212 = t3*t148
    t213 = t3*t149
    t214 = t3*t107*2.0
    t215 = -t180
    t218 = -t183
    t219 = -t144
    t220 = -t196
    t222 = jx*m2*t3*t8*t19*8.0
    t225 = jy*m2*t3*t8*t19*8.0
    t226 = -t200
    t228 = t17*t70*2.0
    t230 = t21*t81
    t232 = t21*t83
    t233 = t21*t93
    t236 = jy*m2*t11*t17*t23*4.0
    t238 = t23*t96
    t239 = -t166
    t241 = -t170
    t242 = -t210
    t243 = -t174
    t247 = -t217
    t250 = t21*t111
    t251 = t21*t112
    t252 = -t221
    t259 = jy*m2*t8*t19*t21*-4.0
    t260 = t21*t128
    t263 = jx*m2*t11*t17*t23*8.0
    t264 = jx*m2*t11*t19*t22*8.0
    t265 = -t235
    t267 = t22*t130
    t268 = jy*m2*t11*t19*t22*8.0
    t269 = t23*t131
    t272 = t17*t150
    t273 = t21*t146
    t275 = t21*t147
    t277 = -t254
    t278 = -t257
    t281 = t30*t76*8.0
    t282 = t30*t77*8.0
    t283 = -t274
    t286 = L*lg*m2*t21*t76*8.0
    t287 = L*lg*m2*t21*t77*8.0
    t57 = -t41
    t61 = -t48
    t114 = -t68
    t115 = -t82
    t123 = -t89
    t129 = -t95
    t135 = -t103
    t136 = t21*t39
    t138 = t21*t41
    t141 = t21*t46
    t143 = t21*t48
    t176 = t3*t108
    t178 = t3*t110
    t191 = -t110
    t197 = t3*t66*2.0
    t199 = t3*t68*2.0
    t204 = t21*t66
    t206 = t21*t68
    t227 = -t203
    t229 = t21*t80
    t231 = t21*t82
    t234 = t23*t94
    t237 = t23*t95
    t245 = -t214
    t253 = -t225
    t256 = -t228
    t261 = t19*t132
    t266 = -t236
    t279 = -t264
    t285 = -t282
    t288 = -t287
    t289 = t30+t49+t75+t77+t145+t153+t219
    t290 = t6+t35+t53+t56+t62+t64+t79+t106+t113+t121+t146+t151+t154+t156+t162+t167+t168+t169+t171+t184+t187+t190+t192+t193+t209+t218+t239+t240+t241+t242+t272+t273+t283+t284
    t223 = -t199
    t246 = -t178
    t258 = t21*t114
    t291 = t29+t31+t51+t72+t78+t92+t104+t108+t109+t111+t124+t126+t149+t152+t158+t160+t163+t164+t175+t176+t177+t179+t181+t182+t188+t189+t191+t194+t195+t213+t215+t216+t243+t244+t245+t246+t247+t248+t250+t276
    t292 = t16+t20+t32+t33+t34+t37+t38+t39+t40+t43+t45+t46+t47+t55+t57+t58+t59+t61+t63+t66+t67+t70+t74+t80+t81+t84+t87+t88+t94+t99+t101+t102+t105+t114+t115+t116+t117+t118+t119+t122+t123+t128+t129+t130+t131+t132+t133+t134+t135+t136+t137+t139+t141+t142+t165+t173+t185+t186+t197+t198+t201+t202+t204+t205+t207+t211+t220+t222+t223+t224+t226+t227+t229+t230+t232+t237+t252+t253+t255+t256+t258+t259+t260+t261+t262+t263+t265+t266+t268+t269+t270+t271+t277+t278+t279+t280+t281+t285+t286+t288
    t293 = 1.0/t292
    ddp = t290*t293*(jy+jx*t19*8.0-jy*np.cos(p*4.0)+jx*t21*t25**2*8.0-jx*t3*t17*1.6e+1+jx*t3*t19*1.6e+1+jy*t3*t17*1.6e+1-jy*t3*t19*1.6e+1+jz*t25*t28*8.0+m2*t11*t28*8.0+jy*t17*t21*t25*8.0+m2*t11*t17*t21*8.0+m2*t11*t21*t25*8.0)*(-1.0/2.0)+t289*t291*t293*4.0+a*input*t293*(jz+jx*t19+jx*t21+jy*t17-jy*t19-jz*t17-jz*t21+m2*t11-jx*t3*t17*2.0+jx*t3*t19*2.0-jx*t17*t21*2.0+jx*t19*t21+jy*t3*t17*2.0-jy*t3*t19*2.0+jy*t17*t21-jy*t19*t21+jz*t17*t21)*4.0
    ddth = t291*t293*(J*4.0+jz*4.0+jy*t28*4.0-jz*t28*4.0+m1*t8+m2*t8*4.0+jx*t25*t28*4.0-jy*t25*t28*4.0+m2*t11*t28*4.0)-t289*t290*t293*4.0+a*input*t289*t293*4.0
    return np.array([dp, dth, ddp, ddth], dtype=float)
