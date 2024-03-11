//Nicholas Christophides 

import java.util.ArrayList;

public class Playlist {
	private String title;
	private ArrayList<Song> songs = new ArrayList<Song>();
	
	public Playlist() {}
	
	public Playlist(String title_) { title = title_; }
	
	public String getTitle() { return title; }
	
	public void setTitle(String title_) { title = title_; }
	
	public void addSong(Song song_) { songs.add(song_); }
	
	public void removeSong(String songName_) {
		for (int i = 0; i < songs.size(); i++) {
			if (songs.get(i).getSongName().equals(songName_)) { songs.remove(i); break; }
		}
	}
	
	public void swapSongs(Song song1, Song song2) {
		if (songs.contains(song1) && songs.contains(song2)) {
			Song temp = songs.get(songs.indexOf(song1));
			int song2Index = songs.indexOf(song2);
			songs.set(songs.indexOf(song1), song2);
			songs.set(song2Index, temp);
		}
	}
	
	public String getDuration() {
		int minutes = 0;
		int seconds = 0;
		for (Song s: songs) {
			minutes += Integer.parseInt(s.getDuration().substring(0, s.getDuration().indexOf(':')));
			seconds += Integer.parseInt(s.getDuration().substring(s.getDuration().indexOf(':')+1));
		}
		minutes += (seconds/60);
		return minutes + " min, " + seconds%60 + " sec";
	}
	
	public void sortSongs() {
		boolean changed;
		do {
			changed = false;
			for (int i = 0; i < songs.size()-1; i++) {
				if (songs.get(i).compareTo(songs.get(i+1)) == -1) {
					changed = true;
					swapSongs(songs.get(i), songs.get(i+1));
				}
			}
		} while (changed);
	}
	
	public String toString() {
		System.out.println(this.getTitle() + "-" + this.getDuration() + "\n");
		for (int i = 0; i < songs.size(); i++) {
			System.out.println(i+1 + ". " + songs.get(i).toString());
		}
		return "";
	}
}
