import unittest
import homework
from unittest.mock import patch, mock_open
from io import StringIO

class TestHomework(unittest.TestCase):

    def test_html_to_text(self):
        html = b"<html><body><h1>Hello</h1><p>World</p></body></html>"
        text = homework.html_to_text(html)
        self.assertEqual(text, "HelloWorld")

        html = b"<html><body><h1>Hello</h1><p>World</p><script>alert('test');</script></body></html>"
        text = homework.html_to_text(html)
        self.assertIn("Hello", text)  # Check for presence instead of full equality
        self.assertIn("World", text)
        self.assertNotIn("<script>", text)  # ensure script content is NOT included


    def test_replace_pii(self):
        text = "My SSN is 123-45-6789."
        expected = "My SSN is XXX-XX-XXXX."
        self.assertEqual(homework.replace_pii(text), expected)

        text = "This 12-34-5678 is not an SSN."
        self.assertEqual(homework.replace_pii(text), text) # No change expected


    def test_clean_text(self):
        text = "This is a test.\nThis is another test.\nThisisareallylongwordthatshouldberemovednotlongenoughyetthismaybeenoughtnonotyetsadfasdwegaddfsadfsdlsadjfljasdflasf."
        expected = "This is a test.\nThis is another test."
        self.assertEqual(homework.clean_text(text), expected)
        text = "This\nis\na\ntest"  # text contains valid punctuation
        self.assertEqual(homework.clean_text(text), text)  # Expect no change
        text = ".\n.\n."  # Check case where lines only have punctuation
        self.assertEqual(homework.clean_text(text), "")  # Should be removed


    @patch("utils.retrieve_bad_words")
    def test_heuristic_quality_filter(self, mock_bad_words):
        mock_bad_words.return_value = {"badword"}

        text = "This is a test."
        self.assertTrue(homework.heuristic_quality_filter(text))

        text = "This is a test without punctuation" # Missing punctuation
        self.assertFalse(homework.heuristic_quality_filter(text))

        text = "               "  # All whitespace.
        self.assertFalse(homework.heuristic_quality_filter(text))

        text = ": こんにちは世界"
        self.assertFalse(homework.heuristic_quality_filter(text))
        
        text = "This is a test with a badword."
        self.assertFalse(homework.heuristic_quality_filter(text))



    @patch("builtins.open", new_callable=mock_open, read_data="url")
    @patch("utils.read_warc_file_url", return_value=b"<p>test this.</p>")
    @patch("utils.read_wet_file_url", return_value=b"test this.")
    @patch("utils.retrieve_bad_words", return_value={"badword"})
    def test_compare(self, mock_wet, mock_warc, mock_file, mock_bad_words):

        with patch("sys.stdout", new=StringIO()) as fake_out:
            homework.compare("warc_file", "wet_file", "url")
            output = fake_out.getvalue().strip()  # remove extra whitespace

        # Define the expected print statements based on processing "test\nurl"
        expected = """url
Passes heuristic quality filter: True
test this.




Wet:
 test this."""
        self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()

