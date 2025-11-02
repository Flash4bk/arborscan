import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class AnalysisResultScreen extends StatefulWidget {
  final String apiUrl;
  final String imageUrl;
  final Map<String, dynamic> result;

  const AnalysisResultScreen({
    super.key,
    required this.apiUrl,
    required this.imageUrl,
    required this.result,
  });

  @override
  State<AnalysisResultScreen> createState() => _AnalysisResultScreenState();
}

class _AnalysisResultScreenState extends State<AnalysisResultScreen> {
  late Map<String, dynamic> result;

  @override
  void initState() {
    super.initState();
    result = widget.result;
  }

  Color _riskColor(String level) {
    switch (level.toLowerCase()) {
      case "высокий":
        return Colors.red;
      case "средний":
        return Colors.orange;
      default:
        return Colors.green;
    }
  }

  @override
  Widget build(BuildContext context) {
    final r = result;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Результат анализа'),
        backgroundColor: Colors.green.shade700,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Вид дерева
            Card(
              elevation: 3,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: ListTile(
                leading: const Icon(Icons.park, color: Colors.green, size: 36),
                title: Text('🌿 Вид: ${r["species"]}', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                subtitle: Text('Уверенность: ${r["confidence"]}%'),
              ),
            ),
            const SizedBox(height: 12),

            // Параметры дерева
            Card(
              elevation: 3,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("📏 Параметры", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    const Divider(),
                    Text("Высота: ${r["height_m"]} м"),
                    Text("Длина кроны: ${r["crown_len_m"]} м"),
                    Text("DBH: ${r["dbh_cm"]} см"),
                    Text("Диаметр ствола: ${r["trunk_diameter_cm"]} см"),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),

            // Погода
            Card(
              elevation: 3,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("🌬️ Погода", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    const Divider(),
                    Text("Ветер: ${r["weather"]["wind"]} м/с"),
                    Text("Порывы: ${r["weather"]["gust"]} м/с"),
                    Text("Температура: ${r["weather"]["temp"]} °C"),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),

            // Почва
            Card(
              elevation: 3,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("🌱 Почва", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    const Divider(),
                    Text("Глина: ${r["soil"]["clay"]}%"),
                    Text("Песок: ${r["soil"]["sand"]}%"),
                    Text("Фактор устойчивости: ${r["soil"]["k_soil"]}"),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),

            // Риск падения
            Card(
              color: _riskColor(r["risk"]["level"]),
              elevation: 3,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("⚠️ Риск падения", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
                    const Divider(color: Colors.white70),
                    Text(
                      "${r["risk"]["level"]} (${r["risk"]["score"].toStringAsFixed(1)}/100)",
                      style: const TextStyle(fontSize: 16, color: Colors.white),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),

            // Фото анализа
            Center(
              child: Column(
                children: [
                  const Text("🖼️ Визуализация анализа", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 10),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.network(
                      widget.imageUrl,
                      fit: BoxFit.contain,
                      loadingBuilder: (context, child, progress) {
                        if (progress == null) return child;
                        return const Padding(
                          padding: EdgeInsets.all(16),
                          child: CircularProgressIndicator(),
                        );
                      },
                      errorBuilder: (context, error, stackTrace) =>
                          const Text("Ошибка загрузки изображения"),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
