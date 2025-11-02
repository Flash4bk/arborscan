import 'dart:async';
import 'package:flutter/material.dart';
import '../main.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnim;
  late Animation<double> _scaleAnim;
  Color _backgroundColor = Colors.green.shade700;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );

    _fadeAnim = CurvedAnimation(parent: _controller, curve: Curves.easeInOut);
    _scaleAnim = Tween<double>(begin: 0.8, end: 1.1)
        .animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));

    _controller.forward();

    // Меняем фон в процессе анимации (эффект "пульсации")
    Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) return;
      setState(() {
        _backgroundColor = _backgroundColor == Colors.green.shade700
            ? Colors.green.shade600
            : Colors.green.shade700;
      });
    });

    // Через 3 секунды переходим к основному экрану
    Timer(const Duration(seconds: 3), () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const TreeAnalyzerScreen()),
      );
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: AnimatedContainer(
        duration: const Duration(seconds: 1),
        color: _backgroundColor,
        child: Stack(
          children: [
            // --- Эффект "плавающих кругов" на фоне ---
            Positioned.fill(
              child: CustomPaint(
                painter: _FloatingCirclesPainter(),
              ),
            ),

            // --- Логотип и текст ---
            Center(
              child: FadeTransition(
                opacity: _fadeAnim,
                child: ScaleTransition(
                  scale: _scaleAnim,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Image.asset(
                        'assets/logo.png',
                        height: 140,
                        width: 140,
                      ),
                      const SizedBox(height: 20),
                      const Text(
                        "ArborScan",
                        style: TextStyle(
                          fontSize: 34,
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1.5,
                        ),
                      ),
                      const SizedBox(height: 40),
                      const Text(
                        "Анализируем природу вокруг вас...",
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 16,
                          fontStyle: FontStyle.italic,
                        ),
                      ),
                      const SizedBox(height: 40),
                      const CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 3,
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// 🎨 Фоновые "плавающие круги" — имитация частиц
class _FloatingCirclesPainter extends CustomPainter {
  final Paint _paint = Paint()..color = Colors.white.withOpacity(0.05);

  @override
  void paint(Canvas canvas, Size size) {
    final now = DateTime.now().millisecondsSinceEpoch / 1000.0;
    for (int i = 0; i < 10; i++) {
      final dx = (size.width / 10) * i + 40 * (i.isEven ? 1 : -1) * (now % 2);
      final dy = (size.height / 10) * (i % 10) + 50 * (i.isOdd ? 1 : -1) * (now % 3);
      final radius = 30 + 10 * (i % 5);
      canvas.drawCircle(Offset(dx % size.width, dy % size.height), radius.toDouble(), _paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
