//Nicholas Christophides 113319835

import java.util.ArrayList;

public class User {
	private String username;
	private ArrayList<Playlist> playlists = new ArrayList<Playlist>();
	
	public User() {}
	
	public User(String username_) { username = username_; }
	
	public String getUsername() { return username; }
	
	public ArrayList<Playlist> getPlaylists() { return playlists; }
	
	public void addPlaylist(Playlist playlist_) { playlists.add(playlist_); }
	
	public void removePlaylist(String title) { 
		for (int i = 0; i < playlists.size(); i++) {
			if (title.equals(playlists.get(i).getTitle())) playlists.remove(i);  
		}
	}
	
	public Playlist getPlaylist(String title) {
		for (int i = 0; i < playlists.size(); i++) {
			if (title.equals(playlists.get(i).getTitle())) return playlists.get(i);
		}
		return null;
	}
	
	void makeCollaborativePlaylist(String playlistTitle, User friend) {
		Playlist playlist = this.getPlaylist(playlistTitle);
		friend.addPlaylist(playlist);
	}
	
	public String toString() {
		return "Username: " + this.getUsername();
	}
}
