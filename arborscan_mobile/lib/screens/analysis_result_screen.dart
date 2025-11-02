import 'package:flutter/material.dart';

class AnalysisResultScreen extends StatelessWidget {
  final Map<String, dynamic> result;
  final String apiUrl;
  final String imageUrl;

  const AnalysisResultScreen({
    super.key,
    required this.result,
    required this.apiUrl,
    required this.imageUrl,
  });

  @override
  Widget build(BuildContext context) {
    final weather = result['weather'];
    final soil = result['soil'];
    final risk = result['risk'];
    final species = result['species'] ?? 'Неизвестно';
    final conf = result['confidence'] ?? 0.0;
    final h = result['height_m'] ?? 0.0;
    final crown = result['crown_len_m'] ?? 0.0;
    final dbh = result['dbh_cm'] ?? 0.0;
    final trunk = result['trunk_diameter_cm'] ?? 0.0;

    final riskColor = {
      'Низкий': Colors.green,
      'Средний': Colors.orange,
      'Высокий': Colors.red,
    }[risk['level']] ?? Colors.grey;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Результат анализа'),
        backgroundColor: Colors.green.shade700,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Вид
            Card(
              elevation: 2,
              child: ListTile(
                leading: const Icon(Icons.nature, color: Colors.green),
                title: Text('Вид: $species'),
                subtitle: Text('Уверенность: ${conf.toStringAsFixed(1)}%'),
              ),
            ),
            const SizedBox(height: 10),

            // Параметры дерева
            Card(
              elevation: 2,
              child: ListTile(
                leading: const Icon(Icons.straighten, color: Colors.green),
                title: const Text('Параметры дерева'),
                subtitle: Text(
                  'Высота: ${h.toStringAsFixed(2)} м\n'
                  'Длина кроны: ${crown.toStringAsFixed(2)} м\n'
                  'Диаметр у земли: ${trunk.toStringAsFixed(1)} см\n'
                  'DBH (на 1.3м): ${dbh.toStringAsFixed(1)} см',
                ),
              ),
            ),
            const SizedBox(height: 10),

            // Погода
            Card(
              elevation: 2,
              child: ListTile(
                leading: const Icon(Icons.cloud, color: Colors.blue),
                title: const Text('Погода'),
                subtitle: weather is Map && weather.containsKey('message')
                    ? Text(weather['message'], style: const TextStyle(color: Colors.grey))
                    : Text(
                        'Скорость ветра: ${weather["wind"] ?? "-"} м/с\n'
                        'Порывы: ${weather["gust"] ?? "-"} м/с\n'
                        'Температура: ${weather["temp"] ?? "-"}°C',
                      ),
              ),
            ),
            const SizedBox(height: 10),

            // Почва
            Card(
              elevation: 2,
              child: ListTile(
                leading: const Icon(Icons.grass, color: Colors.brown),
                title: const Text('Почва'),
                subtitle: soil is Map && soil.containsKey('message')
                    ? Text(soil['message'], style: const TextStyle(color: Colors.grey))
                    : Text(
                        'Глина: ${(soil["clay"] ?? 0).toStringAsFixed(1)}%\n'
                        'Песок: ${(soil["sand"] ?? 0).toStringAsFixed(1)}%\n'
                        'Коэф. устойчивости почвы: ${(soil["k_soil"] ?? 1.0).toStringAsFixed(2)}',
                      ),
              ),
            ),
            const SizedBox(height: 10),

            // Риск
            Card(
              color: riskColor.withOpacity(0.15),
              child: ListTile(
                leading: Icon(Icons.warning_amber_rounded, color: riskColor),
                title: Text('Риск падения: ${risk["level"] ?? "Неизвестно"}'),
                subtitle: Text('Оценка: ${risk["score"] ?? 0}/100'),
              ),
            ),
            const SizedBox(height: 20),

            // Фото
            const Text(
              "📸 Визуализация анализа",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.network(
                imageUrl,
                fit: BoxFit.contain,
                loadingBuilder: (context, child, progress) {
                  if (progress == null) return child;
                  return const Center(child: CircularProgressIndicator());
                },
                errorBuilder: (context, _, __) => Container(
                  color: Colors.grey.shade200,
                  height: 200,
                  child: const Center(child: Text("Ошибка загрузки изображения")),
                ),
              ),
            ),
            const SizedBox(height: 30),

            // Кнопка "Назад"
            ElevatedButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.arrow_back),
              label: const Text("Назад"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green.shade600,
                padding: const EdgeInsets.symmetric(vertical: 14),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
