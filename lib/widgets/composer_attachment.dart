import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';
import 'glass.dart';

/// Thumbnail chip shown above the composer for a pending image attachment.
class ComposerAttachment extends StatelessWidget {
  const ComposerAttachment({
    required this.bytes,
    required this.onRemove,
    super.key,
  });

  final Uint8List bytes;
  final VoidCallback onRemove;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return Glass(
      radius: 18,
      padding: const EdgeInsets.all(8),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: Image.memory(
              bytes,
              width: 56,
              height: 56,
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) => Container(
                width: 56,
                height: 56,
                color: c.surfaceElev,
                child: Icon(Icons.broken_image, color: c.textDim),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Photo attached',
                style: TextStyle(
                  color: c.text,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                'Add a note, then send',
                style: TextStyle(color: c.textDim, fontSize: 11),
              ),
            ],
          ),
          const SizedBox(width: 8),
          GestureDetector(
            onTap: onRemove,
            child: Container(
              width: 28,
              height: 28,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: c.surfaceElev,
                border: Border.all(color: c.border, width: 0.5),
              ),
              child: Icon(Icons.close, color: c.textMuted, size: 16),
            ),
          ),
        ],
      ),
    );
  }
}
