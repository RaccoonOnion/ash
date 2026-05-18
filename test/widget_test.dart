import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';


void main() {
  testWidgets('Placeholder app renders scaffolding text', (WidgetTester tester) async {
    // Re-build the same widget tree that main() creates
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(
        body: Center(child: Text('Ash — scaffolding OK')),
      ),
    ));
    expect(find.text('Ash — scaffolding OK'), findsOneWidget);
  });
}
