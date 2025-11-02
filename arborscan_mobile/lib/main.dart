import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'screens/analysis_result_screen.dart';
import 'screens/splash_screen.dart';


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
      home: const SplashScreen(),
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
  bool _analyzing = false;
  final ImagePicker _picker = ImagePicker();

  // 👉 Укажи адрес своего Railway-сервера:
  final String apiUrl = "https://arborscan-production.up.railway.app";

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() => _image = File(pickedFile.path));
    }
  }

  Future<void> _analyzeTree() async {
    if (_image == null) return;

    setState(() => _analyzing = true);

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiUrl/analyze?lat=55.75&lon=37.62'),
      );
      request.files.add(await http.MultipartFile.fromPath('file', _image!.path));

      var response = await request.send();
      var res = await http.Response.fromStream(response);

      if (res.statusCode == 200) {
        final result = jsonDecode(res.body);
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => AnalysisResultScreen(
              apiUrl: apiUrl,
              imageUrl: "$apiUrl${result["image_path"]}",
              result: result,
            ),
          ),
        );
      } else {
        _showError("Ошибка ${res.statusCode}: ${res.body}");
      }
    } catch (e) {
      _showError("Ошибка отправки: $e");
    }

    setState(() => _analyzing = false);
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message, style: const TextStyle(color: Colors.white)), backgroundColor: Colors.red),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("ArborScan"),
        backgroundColor: Colors.green.shade700,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Center(
          child: _analyzing
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SpinKitFadingCircle(color: Colors.green, size: 70),
                    const SizedBox(height: 20),
                    Text("Идёт анализ дерева...",
                        style: TextStyle(fontSize: 18, color: Colors.green.shade700)),
                  ],
                )
              : Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    if (_image != null)
                      ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Image.file(_image!, height: 250, fit: BoxFit.cover),
                      )
                    else
                      Container(
                        height: 250,
                        width: double.infinity,
                        decoration: BoxDecoration(
                          color: Colors.grey[200],
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Center(
                            child: Text("📸 Загрузите фото дерева",
                                style: TextStyle(fontSize: 18, color: Colors.grey))),
                      ),
                    const SizedBox(height: 20),
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
                    const SizedBox(height: 20),
                    ElevatedButton.icon(
                      onPressed: _image != null ? _analyzeTree : null,
                      icon: const Icon(Icons.analytics),
                      label: const Text("Анализировать"),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green.shade600,
                        minimumSize: const Size(180, 45),
                      ),
                    ),
                  ],
                ),
        ),
      ),
    );
  }
}
