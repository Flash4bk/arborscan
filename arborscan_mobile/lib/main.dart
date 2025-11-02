import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

void main() {
  runApp(const ArborScanApp());
}

class ArborScanApp extends StatelessWidget {
  const ArborScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ArborScan',
      theme: ThemeData(primarySwatch: Colors.green),
      debugShowCheckedModeBanner: false,
      home: const TreeAnalyzerScreen(),
    );
  }
}

class TreeAnalyzerScreen extends StatefulWidget {
  const TreeAnalyzerScreen({super.key});

  @override
  State<TreeAnalyzerScreen> createState() => _TreeAnalyzerScreenState();
}

class _TreeAnalyzerScreenState extends State<TreeAnalyzerScreen> {
  File? _image;
  bool _loading = false;
  Map<String, dynamic>? _result;

  final ImagePicker _picker = ImagePicker();
  final String apiUrl = "https://arborscan-production.up.railway.app/analyze";

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _result = null;
      });
    }
  }

  Future<void> _analyzeImage() async {
    if (_image == null) return;

    setState(() => _loading = true);

    var request = http.MultipartRequest('POST', Uri.parse(apiUrl))
      ..fields['lat'] = '55.75'
      ..fields['lon'] = '37.62'
      ..files.add(await http.MultipartFile.fromPath('file', _image!.path));

    try {
      var response = await request.send();
      var res = await http.Response.fromStream(response);

      if (res.statusCode == 200) {
        setState(() => _result = jsonDecode(res.body));
      } else {
        setState(() => _result = {"error": "Ошибка сервера"});
      }
    } catch (e) {
      setState(() => _result = {"error": e.toString()});
    }

    setState(() => _loading = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('ArborScan')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            if (_image != null)
              Image.file(_image!, height: 200, fit: BoxFit.cover)
            else
              Container(
                height: 200,
                color: Colors.grey[200],
                child: const Center(child: Text("Загрузите фото дерева")),
              ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text("Камера"),
                ),
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo),
                  label: const Text("Галерея"),
                ),
              ],
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _loading ? null : _analyzeImage,
              icon: const Icon(Icons.analytics),
              label: const Text("Анализировать"),
            ),
            const SizedBox(height: 20),
            if (_loading)
              const SpinKitFadingCircle(color: Colors.green)
            else if (_result != null)
              Expanded(
                child: ListView(
                  children: [
                    if (_result!['error'] != null)
                      Text("Ошибка: ${_result!['error']}",
                          style: const TextStyle(color: Colors.red))
                    else
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text("🌳 Вид: ${_result!['species']} "
                              "(${_result!['confidence']}%)"),
                          Text("📏 Высота: ${_result!['height_m']} м"),
                          Text("🌿 Крона: ${_result!['crown_len_m']} м"),
                          Text("🪵 Ствол: ${_result!['dbh_cm']} см"),
                          Text("💨 Ветер: ${_result!['weather']['wind']} м/с"),
                          Text("🌡 Температура: ${_result!['weather']['temp']}°C"),
                          Text("🪴 Почва: k=${_result!['soil']['k_soil']}"),
                          Text("⚠️ Риск: ${_result!['risk']['level']} "
                              "(${_result!['risk']['score']})"),
                        ],
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
