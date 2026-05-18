import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';

/// Glass-styled text input field.
class AshTextField extends StatelessWidget {
  const AshTextField({
    super.key,
    this.hint,
    this.controller,
    this.leadingIcon,
    this.obscureText = false,
    this.onChanged,
    this.onSubmitted,
    this.enabled = true,
    this.height = 52,
    this.borderRadius = 16,
  });

  final String? hint;
  final TextEditingController? controller;
  final IconData? leadingIcon;
  final bool obscureText;
  final ValueChanged<String>? onChanged;
  final ValueChanged<String>? onSubmitted;
  final bool enabled;
  final double height;
  final double borderRadius;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    return SizedBox(
      height: height,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: c.surface,
          borderRadius: BorderRadius.circular(borderRadius),
          border: Border.all(color: c.border, width: 0.5),
        ),
        child: Row(
          children: [
            if (leadingIcon != null)
              Padding(
                padding: const EdgeInsets.only(left: 14),
                child: Icon(
                  leadingIcon,
                  size: 18,
                  color: c.textDim,
                ),
              ),
            Expanded(
              child: TextField(
                controller: controller,
                obscureText: obscureText,
                enabled: enabled,
                onChanged: onChanged,
                onSubmitted: onSubmitted,
                style: TextStyle(
                  color: c.text,
                  fontSize: 15,
                ),
                decoration: InputDecoration(
                  hintText: hint,
                  hintStyle: TextStyle(
                    color: c.textDim,
                    fontSize: 15,
                  ),
                  border: InputBorder.none,
                  contentPadding: EdgeInsets.only(
                    left: leadingIcon != null ? 10 : 16,
                    right: 16,
                  ),
                  isDense: true,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
