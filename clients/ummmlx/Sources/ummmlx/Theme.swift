// SPDX-License-Identifier: Apache-2.0
// Dimensions and neutral/teal palette modified from Voxt ui_tokens.go at commit
// 5645daf0480e9bf572b45a2bba8b8e2f1e642daa. See ThirdParty/Voxt/LICENSE
// and NOTICE. MyGo colors and unrelated voice/provider states were removed.

import SwiftUI

struct Theme {
    let dark: Bool

    private func gray(_ light: UInt32, _ darkValue: UInt32) -> Color {
        Color(hex: dark ? darkValue : light)
    }

    var background: Color { gray(0xFFFFFF, 0x151515) }
    var surface: Color { gray(0xF7F7F7, 0x1C1C1C) }
    var hover: Color { gray(0xF2F2F2, 0x232323) }
    var line: Color { gray(0xEBEBEB, 0x2B2B2B) }
    var text: Color { gray(0x111111, 0xF2F2F2) }
    var secondary: Color { gray(0x6E6E6E, 0xA3A3A3) }
    var tertiary: Color { gray(0xA3A3A3, 0x6B6B6B) }
    var ink: Color { gray(0x111111, 0xF2F2F2) }
    var onInk: Color { gray(0xFFFFFF, 0x151515) }
    var accent: Color { Color(hex: dark ? 0x2BB3A3 : 0x0D9384) }
    var danger: Color { Color(hex: dark ? 0xFF6B47 : 0xEB5230) }

    let controlRadius: CGFloat = 8
    let cardRadius: CGFloat = 12
    let titleSize: CGFloat = 20
    let sectionSize: CGFloat = 15
    let bodySize: CGFloat = 13
    let captionSize: CGFloat = 12
}

private extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: 1
        )
    }
}
